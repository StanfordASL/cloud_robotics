import sys, os
CLOUD_ROOT_DIR=os.environ['CLOUD_ROOT_DIR']
sys.path.append(CLOUD_ROOT_DIR)

RL_ROOT_DIR = CLOUD_ROOT_DIR + '/simulate_RL/'
sys.path.append(RL_ROOT_DIR)

import numpy as np
import tensorflow as tf
from rl_trainer.agents.utils import *
import bisect


class A2CPolicy:
    def __init__(self, n_a, n_s, n_step, n_past, discrete):
        self.n_a = n_a
        self.n_s = n_s
        self.n_step = n_step
        self.n_past = n_past
        self.discrete = discrete
        print('A2C, self.n_a: ', self.n_a)
        print('A2C, self.n_s: ', self.n_s)

    def _build_fc_net(self, h, n_fc, out_type):
        for i, n_fc_cur in enumerate(n_fc):
            fc_cur = out_type + '_fc%d' % i
            h = fc(h, fc_cur, n_fc_cur)
        if out_type == 'pi':
            if self.discrete:
                pi = fc(h, out_type, self.n_a, act=tf.nn.softmax)
                return tf.squeeze(pi)
            else:
                mu = fc(h, 'mu', self.n_a, act=tf.nn.tanh)
                std = fc(h, 'std', self.n_a, act=tf.nn.tanh) + 1
                # need 1e-3 to avoid log_prob explosion
                return [tf.squeeze(mu),  tf.squeeze(std) + 1e-3]
        else:
            v = fc(h, out_type, 1, act=lambda x: x)
            return tf.squeeze(v)

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
        return outs

    def _return_forward_outs(self, out_values):
        if len(out_values) == 1:
            return out_values[0]
        return out_values

    def _discrete_policy_loss(self):
        A_sparse = tf.one_hot(self.A, self.n_a)
        log_pi = tf.log(tf.clip_by_value(self.pi, 1e-10, 1.0))
        entropy = -tf.reduce_sum(self.pi * log_pi, axis=1)
        entropy_loss = -tf.reduce_mean(entropy) * self.entropy_coef
        policy_loss = -tf.reduce_mean(tf.reduce_sum(log_pi * A_sparse, axis=1) * self.ADV)
        return policy_loss, entropy_loss

    def _continuous_policy_loss(self):
        a_norm_dist = tf.contrib.distributions.Normal(self.pi[0], self.pi[1])
        log_prob = a_norm_dist.log_prob(tf.squeeze(self.A, axis=1))
        entropy_loss = -tf.reduce_mean(a_norm_dist.entropy()) * self.entropy_coef
        policy_loss = -tf.reduce_mean(log_prob * self.ADV)
        return policy_loss, entropy_loss

    def prepare_loss(self, optimizer, lr, v_coef, max_grad_norm, alpha, epsilon):
        if not self.discrete:
            self.A = tf.placeholder(tf.float32, [self.n_step, self.n_a])
        else:
            self.A = tf.placeholder(tf.int32, [self.n_step])
        self.ADV = tf.placeholder(tf.float32, [self.n_step])
        self.R = tf.placeholder(tf.float32, [self.n_step])
        self.entropy_coef = tf.placeholder(tf.float32, [])
        if self.discrete:
            policy_loss, entropy_loss = self._discrete_policy_loss()
        else:
            policy_loss, entropy_loss = self._continuous_policy_loss()
        value_loss = tf.reduce_mean(tf.square(self.R - self.v)) * 0.5 * v_coef
        self.loss = policy_loss + value_loss + entropy_loss

        wts = tf.trainable_variables(scope=self.name)
        grads = tf.gradients(self.loss, wts)
        if max_grad_norm > 0:
            grads, self.grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        if optimizer is None:
            # global policy
            self.lr = tf.placeholder(tf.float32, [])
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=alpha,
                                                       epsilon=epsilon)
            self._train = self.optimizer.apply_gradients(list(zip(grads, wts)))
        else:
            # local policy
            self.lr = lr
            self.optimizer = None
            global_name = self.name.split('_')[0] + '_' + str(-1)
            global_wts = tf.trainable_variables(scope=global_name)
            self._train = optimizer.apply_gradients(list(zip(grads, global_wts)))
            self.sync_wt = self._sync_wt(global_wts, wts)
        self.train_out = [entropy_loss, policy_loss, value_loss, self.loss, self.grad_norm]

    def _sample_action(self, pi):
        if self.discrete:
            log_pi = tf.log(tf.clip_by_value(pi, 1e-10, 1.0))
            return tf.cast(tf.squeeze(tf.multinomial([log_pi], 1)), tf.int32)
        else:
            a = pi[0] + pi[1] * tf.random_normal(tf.shape(pi[0]))
            a_clip = tf.clip_by_value(a, -1, 1)
            if self.n_a > 1:
                return a_clip
            return [a_clip]

    @staticmethod
    def _sync_wt(global_wt, local_wt):
        sync_ops = []
        for w1, w2 in zip(global_wt, local_wt):
            sync_ops.append(w2.assign(w1))
        return tf.group(*sync_ops)


class A2CLstmPolicy(A2CPolicy):
    def __init__(self, n_s, n_a, n_step, i_thread, n_past=-1,
                 n_fc=[128], n_lstm=64, discrete=True):
        super().__init__(n_a, n_s, n_step, n_past, discrete)
        self.name = 'a2clstm_' + str(i_thread)
        self.n_lstm = n_lstm
        self.n_fc = n_fc
        self.ob_fw = tf.placeholder(tf.float32, [1, n_s]) # forward 1-step
        self.done_fw = tf.placeholder(tf.float32, [1])
        self.ob_bw = tf.placeholder(tf.float32, [n_step, n_s]) # backward n-step
        self.done_bw = tf.placeholder(tf.float32, [n_step])
        self.states = tf.placeholder(tf.float32, [2, n_lstm * 2])
        with tf.variable_scope(self.name):
            # pi and v use separate nets
            self.pi_fw, pi_state = self._build_net(n_fc, 'forward', 'pi')
            self.v_fw, v_state = self._build_net(n_fc, 'forward', 'v')
            self.a = self._sample_action(self.pi_fw)
            pi_state = tf.expand_dims(pi_state, 0)
            v_state = tf.expand_dims(v_state, 0)
            self.new_states = tf.concat([pi_state, v_state], 0)
        with tf.variable_scope(self.name, reuse=True):
            self.pi, _ = self._build_net(n_fc, 'backward', 'pi')
            self.v, _ = self._build_net(n_fc, 'backward', 'v')
            if i_thread == -1:
                summaries = []
                lstm_pi_w = tf.get_variable('pi_lstm/wx')
                lstm_v_w = tf.get_variable('v_lstm/wx')
                fc_pi_w = tf.get_variable('pi_fc0/w')
                fc_v_w = tf.get_variable('v_fc0/w')
                summaries.append(tf.summary.histogram('wt/lstm_pi', tf.reshape(lstm_pi_w, [-1])))
                summaries.append(tf.summary.histogram('wt/lstm_v', tf.reshape(lstm_v_w, [-1])))
                summaries.append(tf.summary.histogram('wt/fc_pi', tf.reshape(fc_pi_w, [-1])))
                summaries.append(tf.summary.histogram('wt/fc_v', tf.reshape(fc_v_w, [-1])))
                self.summary = tf.summary.merge(summaries)
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

    def backward(self, sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta):
        summary_out = sess.run(self.train_out + [self._train],
                               {self.ob_bw: obs,
                                self.done_bw: dones,
                                self.states: self.states_bw,
                                self.A: acts,
                                self.ADV: Advs,
                                self.R: Rs,
                                self.lr: cur_lr,
                                self.entropy_coef: cur_beta})
        self.states_bw = np.copy(self.states_fw)
        return summary_out[:-1]

    def _backward_policy(self, sess, obs, dones):
        pi = sess.run(self.pi, {self.ob_bw: obs, self.done_bw: dones,
                                self.states: self.states_bw})
        self.states_bw = np.copy(self.states_fw)
        return pi

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
        return outs


class A2CCnn1DPolicy(A2CPolicy):
    def __init__(self, n_s, n_a, n_step, i_thread, n_past=10,
                 n_fc=[128], n_filter=64, m_filter=3, discrete=True):
        super().__init__(n_a, n_s, n_step, n_past, discrete)
        self.name = 'a2ccnn1d_' + str(i_thread)
        self.n_filter = n_filter
        self.m_filter = m_filter
        self.obs = tf.placeholder(tf.float32, [None, n_past, n_s])
        with tf.variable_scope(self.name):
            # pi and v use separate nets 
            self.pi = self._build_net(n_fc, 'pi')
            self.v = self._build_net(n_fc, 'v')
            self.a = self._sample_action(self.pi)
        if i_thread == -1:
            with tf.variable_scope(self.name, reuse=True):
                summaries = []
                conv1_pi_w = tf.get_variable('pi_conv1/w')
                conv1_v_w = tf.get_variable('v_conv1/w')
                fc_pi_w = tf.get_variable('pi_fc0/w')
                fc_v_w = tf.get_variable('v_fc0/w')
                summaries.append(tf.summary.histogram('wt/conv1_pi', tf.reshape(conv1_pi_w, [-1])))
                summaries.append(tf.summary.histogram('wt/conv1_v', tf.reshape(conv1_v_w, [-1])))
                summaries.append(tf.summary.histogram('wt/fc_pi', tf.reshape(fc_pi_w, [-1])))
                summaries.append(tf.summary.histogram('wt/fc_v', tf.reshape(fc_v_w, [-1])))
                self.summary = tf.summary.merge(summaries)
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

    def backward(self, sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta):
        obs = self._recent_ob(np.array(obs), np.array(dones),  ob_type='backward')
        summary_out = sess.run(self.train_out + [self._train],
                               {self.obs: obs,
                                self.A: acts,
                                self.ADV: Advs,
                                self.R: Rs,
                                self.lr: cur_lr,
                                self.entropy_coef: cur_beta})
        return summary_out[:-1]

    def _backward_policy(self, sess, obs, dones):
        obs = self._recent_ob(np.array(obs), np.array(dones),  ob_type='backward')
        print(obs)
        return sess.run(self.pi, {self.obs: obs})


def test_forward_backward_policies(sess, policy, x, done):
    n_step = len(done)
    for i in range(n_step):
        print('forward')
        pi = policy.forward(sess, x[i], done[i], out_type='p')
        print(pi)
        print('-' * 40)
    print('backward')
    pi = policy._backward_policy(sess, x, done)
    print(pi)
    print('=' * 40)


def test_policies():
    sess = tf.Session()
    n_s, n_a, n_step, n_past = 3, 4, 5, 8
    p_lstm = A2CLstmPolicy(n_s, n_a, n_step, n_lstm=5)
    p_cnn1 = A2CCnn1DPolicy(n_s, n_a, n_step, n_past)
    sess.run(tf.global_variables_initializer())
    print('=' * 16 + 'first batch' + '=' * 16)
    x = np.random.randn(n_step, n_s)
    done = np.array([0,0,0,1,0])
    x[3,:] = 0
    print('LSTM:')
    test_forward_backward_policies(sess, p_lstm, x, done)
    print('CNN1D:')
    test_forward_backward_policies(sess, p_cnn1, x, done)
    print('=' * 16 + 'second batch' + '=' * 16)
    x = np.random.randn(n_step, n_s)
    done = np.array([0,1,1,0,0])
    x[1,:] = 0
    print('LSTM:')
    test_forward_backward_policies(sess, p_lstm, x, done)
    print('CNN1D:')
    test_forward_backward_policies(sess, p_cnn1, x, done)


if __name__ == '__main__':
    test_policies()
