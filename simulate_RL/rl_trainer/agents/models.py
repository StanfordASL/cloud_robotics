import sys, os
CLOUD_ROOT_DIR=os.environ['CLOUD_ROOT_DIR']
sys.path.append(CLOUD_ROOT_DIR)

RL_ROOT_DIR = CLOUD_ROOT_DIR + '/simulate_RL/'
sys.path.append(RL_ROOT_DIR)

from rl_trainer.agents.utils import *
from rl_trainer.agents.a2c_policies import *
from rl_trainer.agents.ppo_policies import *


class A2C:
    def __init__(self, sess, n_s, n_a, total_step, i_thread=-1, optimizer=None, lr=None,
                 model_config=None, discrete=True):
        self.name = 'a2c'
        self.sess = sess
        self.i_thread = i_thread
        self.total_step = total_step
        self._init_policy(n_s, n_a, model_config, discrete)
        self.reward_norm = model_config.getfloat('REWARD_NORM')
        self.discrete = discrete
        if total_step > 0:
            # global lr and entropy beta scheduler
            if i_thread == -1:
                self.lr_scheduler = self._init_scheduler(model_config)
                self.beta_scheduler = self._init_scheduler(model_config, name='ENTROPY')
            self._init_train(optimizer, lr, model_config)

    def init_train(self):
        pass

    def save(self, saver, model_dir, global_step):
        saver.save(self.sess, model_dir, global_step=global_step)

    def load(self, saver, model_dir, checkpoint=None):
        if self.i_thread == -1:
            save_file = None
            save_step = 0
            if os.path.exists(model_dir):
                if checkpoint is None:
                    for file in os.listdir(model_dir):
                        if file.startswith('checkpoint'):
                            prefix = file.split('.')[0]
                            tokens = prefix.split('-')
                            if len(tokens) != 2:
                                continue
                            cur_step = int(tokens[1])
                            if cur_step > save_step:
                                save_file = prefix
                                save_step = cur_step
                else:
                    save_file = 'checkpoint-' + str(int(checkpoint))
            if save_file is not None:
                saver.restore(self.sess, model_dir + save_file)
                print('checkpoint loaded: ', save_file)
            else:
                print('could not find old checkpoint')

    def backward(self, R, cur_lr, cur_beta):
        obs, acts, dones, Rs, Advs = self.trans_buffer.sample_transition(R, self.discrete)
        return self.policy.backward(self.sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta)

    def forward(self, ob, done, mode='pv'):
        return self.policy.forward(self.sess, ob, done, mode)

    def add_transition(self, ob, action, reward, value, done):
        if self.reward_norm:
            reward /= self.reward_norm
        self.trans_buffer.add_transition(ob, action, reward, value, done)

    def _init_policy(self, n_s, n_a, model_config, discrete):
        policy_name = model_config.get('POLICY')
        n_step = model_config.getint('NUM_STEP')
        n_past = model_config.getint('NUM_PAST')
        n_fc = model_config.get('NUM_FC').split(',')
        n_fc = [int(x) for x in n_fc]
        self.n_step = n_step
        if policy_name == 'lstm':
            n_lstm = model_config.getint('NUM_LSTM')
            self.policy = A2CLstmPolicy(n_s, n_a, n_step, self.i_thread, n_past, n_fc=n_fc,
                                        n_lstm=n_lstm, discrete=discrete)
        elif policy_name == 'cnn1':
            n_filter = model_config.getint('NUM_FILTER')
            m_filter = model_config.getint('SIZE_FILTER')
            self.policy = A2CCnn1DPolicy(n_s, n_a, n_step, self.i_thread, n_past,
                                         n_fc=n_fc, n_filter=n_filter,
                                         m_filter=m_filter, discrete=discrete)

    def _init_scheduler(self, model_config, name='LR'):
        val_init = model_config.getfloat(name + '_INIT')
        val_decay = model_config.get(name + '_DECAY')
        if val_decay == 'constant':
            return Scheduler(val_init, decay=val_decay)
        if name + '_MIN' in model_config:
            val_min = model_config.getfloat(name + '_MIN')
        else:
            val_min = 0
        decay_step = self.total_step
        if name + '_RATIO' in model_config:
            decay_step *= model_config.getfloat(name + '_RATIO')
        return Scheduler(val_init, val_min=val_min, total_step=decay_step, decay=val_decay)

    def _init_train(self, optimizer, lr, model_config):
        v_coef = model_config.getfloat('VALUE_COEF')
        max_grad_norm = model_config.getfloat('MAX_GRAD_NORM')
        alpha = model_config.getfloat('RMSP_ALPHA')
        epsilon = model_config.getfloat('RMSP_EPSILON')
        gamma = model_config.getfloat('GAMMA')
        self.policy.prepare_loss(optimizer, lr, v_coef, max_grad_norm, alpha, epsilon)
        self.trans_buffer = OnPolicyBuffer(gamma)
        if self.i_thread == -1:
            self.optimizer = self.policy.optimizer
            self.lr = self.policy.lr


class PPO(A2C):
    def __init__(self, sess, n_s, n_a, total_step, i_thread=-1, optimizer=None, lr=None,
                 clip=None, model_config=None, discrete=True):
        self.name = 'ppo'
        self.sess = sess
        self.i_thread = i_thread
        self.total_step = total_step
        self._init_policy(n_s, n_a, model_config, discrete)
        self.reward_norm = model_config.getfloat('REWARD_NORM')
        self.discrete = discrete
        if total_step > 0:
            # global lr and entropy beta scheduler
            if i_thread == -1:
                self.lr_scheduler = self._init_scheduler(model_config)
                self.beta_scheduler = self._init_scheduler(model_config, name='ENTROPY')
                self.clip_scheduler = self._init_scheduler(model_config, name='CLIP')
            self._init_train(optimizer, lr, clip, model_config)

    def backward(self, R, cur_lr, cur_beta, cur_clip):
        # TODO: add epoch and minibatch
        obs, acts, dones, Rs, Advs, Vs, Logprobs = self.trans_buffer.sample_transition(R, self.discrete)
        n_update = int(self.n_step / self.n_batch)
        summary = []
        for _ in range(self.n_epoch):
            inds = np.random.permutation(self.n_step)
            for i in range(n_update):
                cur_inds = inds[(i * self.n_batch):((i + 1) * self.n_batch)]
                val = self.policy.backward(self.sess, obs[cur_inds], acts[cur_inds], dones[cur_inds],
                                           Rs[cur_inds], Advs[cur_inds], Vs[cur_inds], Logprobs[cur_inds],
                                           cur_lr, cur_beta, cur_clip)
                summary.append(val)
        return list(np.mean(np.array(summary), axis=0))

    def forward(self, ob, done, mode='pv'):
        return self.policy.forward(self.sess, ob, done, mode)

    def add_transition(self, ob, action, reward, value, done, logprob):
        if self.reward_norm:
            reward /= self.reward_norm
        self.trans_buffer.add_transition(ob, action, reward, value, done, logprob)

    def _init_policy(self, n_s, n_a, model_config, discrete):
        policy_name = model_config.get('POLICY')
        n_step = model_config.getint('NUM_STEP')
        n_past = model_config.getint('NUM_PAST')
        n_fc = model_config.get('NUM_FC').split(',')
        n_fc = [int(x) for x in n_fc]
        self.n_step = n_step
        n_batch = model_config.getint('BATCH_SIZE')
        self.n_batch = n_batch
        if policy_name == 'lstm':
            n_lstm = model_config.getint('NUM_LSTM')
            self.policy = PPOLstmPolicy(n_s, n_a, n_batch, self.i_thread, n_past, n_fc=n_fc,
                                        n_lstm=n_lstm, discrete=discrete)
        elif policy_name == 'cnn1':
            n_filter = model_config.getint('NUM_FILTER')
            m_filter = model_config.getint('SIZE_FILTER')
            self.policy = PPOCnn1DPolicy(n_s, n_a, n_batch, self.i_thread, n_past,
                                         n_fc=n_fc, n_filter=n_filter,
                                         m_filter=m_filter, discrete=discrete)

    def _init_train(self, optimizer, lr, clip, model_config):
        v_coef = model_config.getfloat('VALUE_COEF')
        max_grad_norm = model_config.getfloat('MAX_GRAD_NORM')
        alpha = model_config.getfloat('RMSP_ALPHA')
        epsilon = model_config.getfloat('RMSP_EPSILON')
        gamma = model_config.getfloat('GAMMA')
        self.n_epoch = model_config.getint('NUM_EPOCH')
        self.policy.prepare_loss(optimizer, lr, v_coef, max_grad_norm, alpha, epsilon, clip)
        self.trans_buffer = PPOBuffer(gamma)
        if self.i_thread == -1:
            self.optimizer = self.policy.optimizer
            self.lr = self.policy.lr
            self.clip = self.policy.clip

