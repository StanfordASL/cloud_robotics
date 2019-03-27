import os
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_color_codes()


class GlobalCounter:
    def __init__(self, total_step, save_step, log_step):
        self.counter = itertools.count(1)
        self.cur_step = 0
        self.cur_save_step = 0
        self.total_step = total_step
        self.save_step = save_step
        self.log_step = log_step

    def next(self):
        self.cur_step = next(self.counter)
        return self.cur_step

    def should_save(self):
        save = False
        if (self.cur_step - self.cur_save_step) >= self.save_step:
            save = True
            self.cur_save_step = self.cur_step
        return save

    def should_log(self):
        return (self.cur_step % self.log_step == 0)

    def should_stop(self):
        return (self.cur_step >= self.total_step)


def signal_handler(signal, frame):
    print('You pressed Ctrl+C!')


def init_out_dir(base_dir, mode):
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    save_path = base_dir + '/model/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if mode == 'train':
        log_path = base_dir + '/log/'
        if not os.path.exists(log_path):
            os.mkdir(log_path)
    elif mode == 'evaluate':
        log_path = base_dir + '/evaluate/'
        if not os.path.exists(log_path):
            os.mkdir(log_path)
    return save_path, log_path


def init_model_summary(algo):
    policy_loss = tf.placeholder(tf.float32, [])
    value_loss = tf.placeholder(tf.float32, [])
    total_loss = tf.placeholder(tf.float32, [])
    lr = tf.placeholder(tf.float32, [])
    gradnorm = tf.placeholder(tf.float32, [])
    if algo in ['a2c', 'ppo']:
        entropy_loss = tf.placeholder(tf.float32, [])
        beta = tf.placeholder(tf.float32, [])
        if algo == 'ppo':
            policy_kl = tf.placeholder(tf.float32, [])
            clip_rate = tf.placeholder(tf.float32, [])
    summaries = []
    summaries.append(tf.summary.scalar('loss/policy', policy_loss))
    summaries.append(tf.summary.scalar('loss/value', value_loss))
    summaries.append(tf.summary.scalar('loss/total', total_loss))
    summaries.append(tf.summary.scalar('train/lr', lr))
    summaries.append(tf.summary.scalar('train/gradnorm', gradnorm))
    if algo in ['a2c', 'ppo']:
        summaries.append(tf.summary.scalar('loss/entropy', entropy_loss))
        summaries.append(tf.summary.scalar('train/beta', beta))
        if algo == 'a2c':
            summary = tf.summary.merge(summaries)
            return (summary, policy_loss, value_loss,
                    total_loss, lr, gradnorm, entropy_loss, beta)
        summaries.append(tf.summary.scalar('train/policy_kl', policy_kl))
        summaries.append(tf.summary.scalar('train/clip_rate', clip_rate))
        summary = tf.summary.merge(summaries)
        return (summary, policy_loss, value_loss,
                total_loss, lr, gradnorm, entropy_loss,
                beta, policy_kl, clip_rate)


def plot_episode(actions, states, rewards, run, plot_path):
    fig = plt.figure(figsize=(12, 18))
    title = fig.suptitle('EPISODE RUN: %d' % run, fontsize='x-large')
    plt.subplot(3, 1, 1)
    for i in range(states.shape[1]):
        plt.plot(states[:,i], 'o-', markersize=6, markeredgewidth=0,
                 label=('state_%d' % i))
    plt.legend(fontsize=15, loc='best')
    plt.yticks(fontsize=15)
    plt.ylabel('Normalized states', fontsize=15)

    plt.subplot(3, 1, 2)
    plt.plot(actions, 'o-', markersize=12, markeredgewidth=0, linewidth=3)
    plt.ylabel('Actions', fontsize=15)
    plt.yticks(fontsize=15)

    plt.subplot(3, 1, 3)
    plt.plot(rewards, 'o-', markersize=12, markeredgewidth=0, linewidth=3)
    plt.ylabel('Rewards', fontsize=15)
    plt.yticks(fontsize=15)

    fig.tight_layout()
    title.set_y(0.95)
    fig.subplots_adjust(top=0.9)
    fig.savefig(plot_path + '/RUN' + str(run))
