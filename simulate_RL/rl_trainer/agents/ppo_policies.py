import sys, os
CLOUD_ROOT_DIR=os.environ['CLOUD_ROOT_DIR']
sys.path.append(CLOUD_ROOT_DIR)

RL_ROOT_DIR = CLOUD_ROOT_DIR + '/simulate_RL/'
sys.path.append(RL_ROOT_DIR)

import numpy as np
import tensorflow as tf
from rl_trainer.agents.utils import *
from rl_trainer.agents.a2c_policies import A2CPolicy
import bisect

class PPOPolicy(A2CPolicy):
    def __init__(self, n_a, n_s, n_step, n_past, discrete):
        super().__init__(n_a, n_s, n_step, n_past, discrete)

    def prepare_loss(self, optimizer, lr, v_coef, max_grad_norm, alpha, epsilon,
                     clip):
        if optimizer is None:
            # global policy
            self.lr = tf.placeholder(tf.float32, [])
            self.clip = tf.placeholder(tf.float32, [])
        else:
            self.lr = lr
            self.clip = clip
        self.ADV = tf.placeholder(tf.float32, [self.n_step])
        self.R = tf.placeholder(tf.float32, [self.n_step])
        self.entropy_coef = tf.placeholder(tf.float32, [])
        # old v and pi values for clipping
        self.OLDV = tf.placeholder(tf.float32, [self.n_step])
        self.OLDLOGPROB = tf.placeholder(tf.float32, [self.n_step])
        policy_loss, entropy_loss = self._policy_loss()
        value_loss = self._value_loss() * v_coef
        self.loss = policy_loss + value_loss + entropy_loss

        wts = tf.trainable_variables(scope=self.name)
        grads = tf.gradients(self.loss, wts)
        if max_grad_norm > 0:
            grads, self.grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        if optimizer is None:
            # global policy
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=alpha,
                                                       epsilon=epsilon)
            self._train = self.optimizer.apply_gradients(list(zip(grads, wts)))
        else:
            # local policy
            self.optimizer = None
            global_name = self.name.split('_')[0] + '_' + str(-1)
            global_wts = tf.trainable_variables(scope=global_name)
            self._train = optimizer.apply_gradients(list(zip(grads, global_wts)))
            self.sync_wt = self._sync_wt(global_wts, wts)
        self.train_out = [entropy_loss, policy_loss, value_loss, self.loss, self.grad_norm,
                          self.policy_kl, self.clip_rate]

    def _get_forward_outs(self, out_type):
        outs = []
        if 'p' in out_type:
            outs.append(self.a)
            if self.discrete:
                outs.append(self.pi)
            else:
                outs += self.pi
        if 'v' in out_type:
            outs.append(self.v)
        if ('p' in out_type) and ('v' in out_type):
            outs.append(self.logprob)
        return outs

    def _get_logprob(self, pi, A):
        if self.discrete:
            A_sparse = tf.one_hot(A, self.n_a)
            log_pi = tf.log(tf.clip_by_value(pi, 1e-10, 1.0))
            log_prob = tf.reduce_sum(log_pi * A_sparse, axis=1)
            entropy = -tf.reduce_sum(pi * log_pi, axis=1)
        else:
            a_norm_dist = tf.contrib.distributions.Normal(pi[0], pi[1])
            log_prob = a_norm_dist.log_prob(tf.squeeze(A, axis=1))
            entropy = a_norm_dist.entropy()
        return tf.squeeze(log_prob), entropy

    def _policy_loss(self):
        logprob, entropy = self._get_logprob(self.pi, self.A)
        entropy_loss = -tf.reduce_mean(entropy) * self.entropy_coef
        ratio = tf.exp(logprob - self.OLDLOGPROB)
        ratio_clip = tf.clip_by_value(ratio, 1 - self.clip, 1 + self.clip)
        policy_loss = -tf.reduce_mean(tf.maximum(ratio, ratio_clip) * self.ADV)
        self.policy_kl = .5 * tf.reduce_mean(tf.square(logprob - self.OLDLOGPROB))
        self.clip_rate = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), self.clip)))
        return policy_loss, entropy_loss

    def _value_loss(self):
        # TODO: better usage on v_old
        # v_clip = self.OLDV + tf.clip_by_value(self.v - self.OLDV, -self.clip * self.OLDV,
        #                                       self.clip * self.OLDV)
        v_loss = tf.square(self.R - self.v)
        # v_loss_clip = tf.square(self.R - v_clip)
        return tf.reduce_mean(v_loss) * 0.5


class PPOLstmPolicy(PPOPolicy):
    def __init__(self, n_s, n_a, n_step, i_thread, n_past=-1,
                 n_fc=[128], n_lstm=64, discrete=True):
        super().__init__(n_a, n_s, n_step, n_past, discrete)
        self.name = 'ppolstm_' + str(i_thread)
        self.n_lstm = n_lstm
        self.n_fc = n_fc
        self.ob_fw = tf.placeholder(tf.float32, [1, n_s]) # forward 1-step
        self.done_fw = tf.placeholder(tf.float32, [1])
        self.ob_bw = tf.placeholder(tf.float32, [n_step, n_s]) # backward n-step
        self.done_bw = tf.placeholder(tf.float32, [n_step])
        self.states = tf.placeholder(tf.float32, [2, n_lstm * 2])
        if not self.discrete:
            self.A = tf.placeholder(tf.float32, [n_step, n_a])
        else:
            self.A = tf.placeholder(tf.int32, [n_step])
        with tf.variable_scope(self.name):
            # pi and v use separate nets
            self.pi_fw, pi_state = self._build_net(n_fc, 'forward', 'pi')
            self.v_fw, v_state = self._build_net(n_fc, 'forward', 'v')
            self.a = self._sample_action(self.pi_fw)
            self.logprob, _ = self._get_logprob(self.pi_fw, tf.expand_dims(self.a, 0))
            pi_state = tf.expand_dims(pi_state, 0)
            v_state = tf.expand_dims(v_state, 0)
            self.new_states = tf.concat([pi_state, v_state], 0)
        with tf.variable_scope(self.name, reuse=True):
            self.pi, _ = self._build_net(n_fc, 'backward', 'pi')
            self.v, _ = self._build_net(n_fc, 'backward', 'v')
        self._reset()

    def _build_net(self, n_fc, in_type, out_type):
        if in_type == 'forward':
            ob = self.ob_fw
            done = self.done_fw
        else:
            ob = self.ob_bw
            done = self.done_bw
        if out_type == 'pi':
            states = self.states[0]
        else:
            states = self.states[1]
        h, new_states = lstm(ob, done, states, out_type + '_lstm')
        out_val = self._build_fc_net(h, n_fc, out_type)
        return out_val, new_states

    def _reset(self):
        # forget the cumulative states every cum_step
        self.states_fw = np.zeros((2, self.n_lstm * 2), dtype=np.float32)
        self.states_bw = np.zeros((2, self.n_lstm * 2), dtype=np.float32)
        self.cur_step = 0

    def forward(self, sess, ob, done, out_type='pv'):
        outs = self._get_forward_outs(out_type)
        # update state only when p is called
        if 'p' in out_type:
            outs.append(self.new_states)
        out_values = sess.run(outs, {self.ob_fw: np.array([ob]),
                                     self.done_fw: np.array([done]),
                                     self.states: self.states_fw})
        if 'p' in out_type:
            self.states_fw = out_values[-1]
            out_values = out_values[:-1]
        if done:
            self.cur_step = 0
        self.cur_step += 1
        if (self.n_past > 0) and (self.cur_step >= self.n_past):
            self.states_fw = np.zeros((2, self.n_lstm * 2), dtype=np.float32)
            self.cur_step = 0
        return self._return_forward_outs(out_values)

    def backward(self, sess, obs, acts, dones, Rs, Advs, Vs, Logprobs,
                 cur_lr, cur_beta, cur_clip):
        summary_out = sess.run(self.train_out + [self._train],
                               {self.ob_bw: obs,
                                self.done_bw: dones,
                                self.states: self.states_bw,
                                self.A: acts,
                                self.ADV: Advs,
                                self.R: Rs,
                                self.OLDV: Vs,
                                self.OLDLOGPROB: Logprobs,
                                self.lr: cur_lr,
                                self.entropy_coef: cur_beta,
                                self.clip: cur_clip})
        self.states_bw = np.copy(self.states_fw)
        return summary_out[:-1]

    def _get_forward_outs(self, out_type):
        outs = []
        if 'p' in out_type:
            outs.append(self.a)
            if self.discrete:
                outs.append(self.pi_fw)
            else:
                outs += self.pi_fw
        if 'v' in out_type:
            outs.append(self.v_fw)
        if ('p' in out_type) and ('v' in out_type):
            outs.append(self.logprob)
        return outs


class PPOCnn1DPolicy(PPOPolicy):
    def __init__(self, n_s, n_a, n_step, i_thread, n_past=10,
                 n_fc=[128], n_filter=64, m_filter=3, discrete=True):
        super().__init__(n_a, n_s, n_step, n_past, discrete)
        self.name = 'ppocnn1d_' + str(i_thread)
        self.n_filter = n_filter
        self.m_filter = m_filter
        self.obs = tf.placeholder(tf.float32, [None, n_past, n_s])
        if not self.discrete:
            self.A = tf.placeholder(tf.float32, [None, n_a])
        else:
            self.A = tf.placeholder(tf.int32, [None])
        with tf.variable_scope(self.name):
            # pi and v use separate nets
            self.pi = self._build_net(n_fc, 'pi')
            self.v = self._build_net(n_fc, 'v')
            self.a = self._sample_action(self.pi)
            self.logprob, _ = self._get_logprob(self.pi, tf.expand_dims(self.a, 0))
        self._reset()

    def _build_net(self, n_fc, out_type):
        h = conv(self.obs, out_type + '_conv1', self.n_filter, self.m_filter, conv_dim=1)
        n_conv_fc = np.prod([v.value for v in h.shape[1:]])
        h = tf.reshape(h, [-1, n_conv_fc])
        return self._build_fc_net(h, n_fc, out_type)

    def _reset(self):
        self.recent_obs_fw = np.zeros((self.n_past-1, self.n_s))
        self.recent_obs_bw = np.zeros((self.n_past-1, self.n_s))
        self.recent_dones_fw = np.zeros(self.n_past-1)
        self.recent_dones_bw = np.zeros(self.n_past-1)

    def _recent_ob(self, obs, dones, ob_type='forward'):
        # convert [n_step, n_s] to [n_step, n_past, n_s]
        num_obs = len(obs)
        if ob_type == 'forward':
            recent_obs = np.copy(self.recent_obs_fw)
            recent_dones = np.copy(self.recent_dones_fw)
        else:
            recent_obs = np.copy(self.recent_obs_bw)
            recent_dones = np.copy(self.recent_dones_bw)
        comb_obs = np.vstack([recent_obs, obs])
        comb_dones = np.concatenate([recent_dones, dones])
        new_obs = []
        inds = list(np.nonzero(comb_dones)[0])
        for i in range(num_obs):
            cur_obs = np.copy(comb_obs[i:(i + self.n_past)])
            # print(cur_obs)
            if len(inds):
                k = bisect.bisect_left(inds, (i + self.n_past)) - 1
                if (k >= 0) and (inds[k] > i):
                    cur_obs[:(int(inds[k]) - i)] *= 0
            new_obs.append(cur_obs)
        recent_obs = comb_obs[(1-self.n_past):]
        recent_dones = comb_dones[(1-self.n_past):]
        if ob_type == 'forward':
            self.recent_obs_fw = recent_obs
            self.recent_dones_fw = recent_dones
        else:
            self.recent_obs_bw = recent_obs
            self.recent_dones_bw = recent_dones
        return np.array(new_obs)

    def forward(self, sess, ob, done, out_type='pv'):
        ob = self._recent_ob(np.array([ob]), np.array([done]))
        outs = self._get_forward_outs(out_type)
        out_values = sess.run(outs, {self.obs: ob})
        return self._return_forward_outs(out_values)

    def backward(self, sess, obs, acts, dones, Rs, Advs, Vs, Logprobs,
                 cur_lr, cur_beta, cur_clip):
        obs = self._recent_ob(np.array(obs), np.array(dones),  ob_type='backward')
        summary_out = sess.run(self.train_out + [self._train],
                               {self.obs: obs,
                                self.A: acts,
                                self.ADV: Advs,
                                self.R: Rs,
                                self.OLDV: Vs,
                                self.OLDLOGPROB: Logprobs,
                                self.lr: cur_lr,
                                self.entropy_coef: cur_beta,
                                self.clip: cur_clip})
        return summary_out[:-1]
