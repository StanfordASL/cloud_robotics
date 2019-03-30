#!/usr/bin/env python3

import sys, os
CLOUD_ROOT_DIR=os.environ['CLOUD_ROOT_DIR']
sys.path.append(CLOUD_ROOT_DIR)

RL_ROOT_DIR = CLOUD_ROOT_DIR + '/simulate_RL/'
sys.path.append(RL_ROOT_DIR)

import argparse
import configparser
import signal
import tensorflow as tf
import threading

from rl_trainer.agents.models import A2C, PPO
#from stochastic_simulator.stochastic_video_offload_env import StochasticInputOffloadEnv
from rl_trainer.train import Trainer, AsyncTrainer, Evaluator
from rl_trainer.utils import GlobalCounter, init_out_dir, init_model_summary, signal_handler

CLOUD_ROOT_DIR = os.environ['CLOUD_ROOT_DIR']

#sys.path.append(CLOUD_ROOT_DIR + '/FourActionSimulator/') 
#from four_action_simulator_v1 import FourActionOffloadEnv

sys.path.append(CLOUD_ROOT_DIR + '/simulate_RL/FaceNet_four_action_simulator/')
from four_action_simulator_v1_fnet import FourActionOffloadEnv

def parse_args():
    default_config_path = os.path.join(CLOUD_ROOT_DIR, 'rl_configs/test.ini')
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, required=False,
                        default=CLOUD_ROOT_DIR, help="base directory")
    parser.add_argument('--config-path', type=str, required=False,
                        default=default_config_path, help="config path")
    parser.add_argument('--mode', type=str, required=False,
                        default='train', help="train or evaluate")
    parser.add_argument('--agent', type=str, required=False,
                        default='a2c', help="a2c, ppo")
    parser.add_argument('--test-seeds', type=str, required=False,
                        default='100,200', help="seeds to test a pre-trained RL model")
    parser.add_argument('--train-seeds', type=str, required=False,
                        default='False', help="seeds to test a pre-trained RL model")
    parser.add_argument('--env-name', type=str, required=False,
                        default='stochastic', help="toggle between environments")
    return parser.parse_args()


def train(parser, train_seeds, test_seeds, algo, env_name, base_dir):
    seed = parser.getint('TRAIN_CONFIG', 'SEED')
    num_env = parser.getint('TRAIN_CONFIG', 'NUM_ENV')
    #fraction = parser.getfloat('ENV_CONFIG', 'QUARY_BUDGET_FRACTION')

    if env_name == 'stochastic':
        env = StochasticInputOffloadEnv(query_budget_frac=fraction)
    elif env_name == 'AQE':
        env = AlwaysQueryEdgeOffloadEnv()
    elif env_name == 'FourAction':
        env = FourActionOffloadEnv()
    else:
        pass

    env._seed(seed)
    n_a = env.n_a
    n_s = env.n_s
    print('n_a: ', n_a)
    print('n_s: ', n_s)
    total_step = int(parser.getfloat('TRAIN_CONFIG', 'MAX_STEP'))
    #base_dir = parser.get('TRAIN_CONFIG', 'BASE_DIR')
    save_step = int(parser.getfloat('TRAIN_CONFIG', 'SAVE_INTERVAL'))
    log_step = int(parser.getfloat('TRAIN_CONFIG', 'LOG_INTERVAL'))
    save_path, log_path = init_out_dir(base_dir, 'train')

    tf.set_random_seed(seed)
    config = tf.ConfigProto(allow_soft_placement=True)
    # key statement added here for working on the jetson tx2 GPU
    # https://devtalk.nvidia.com/default/topic/1029742/jetson-tx2/tensorflow-1-6-not-working-with-jetpack-3-2/
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    if algo == 'a2c':
        global_model = A2C(sess, n_s, n_a, total_step, model_config=parser['MODEL_CONFIG'],
                           discrete=True)
    elif algo == 'ppo':
        global_model = PPO(sess, n_s, n_a, total_step, model_config=parser['MODEL_CONFIG'],
                           discrete=True)
    else:
        global_model = None
    global_counter = GlobalCounter(total_step, save_step, log_step)
    coord = tf.train.Coordinator()
    threads = []
    trainers = []
    model_summary = init_model_summary(global_model.name)

    if num_env == 1:
        # regular training
        summary_writer = tf.summary.FileWriter(log_path, sess.graph)
        print('train_seeds: ', train_seeds)
        trainer = Trainer(env, global_model, save_path, summary_writer, global_counter, model_summary, train_seeds = train_seeds)
        trainers.append(trainer)
    else:
        assert(algo in ['a2c', 'ppo'])
        # asynchronous training
        lr_scheduler = global_model.lr_scheduler
        beta_scheduler = global_model.beta_scheduler
        optimizer = global_model.optimizer
        lr = global_model.lr
        clip_scheduler = None
        if algo == 'ppo':
            clip = global_model.clip
            clip_scheduler = global_model.clip_scheduler
        wt_summary = None
        reward_summary = None
        summary_writer = tf.summary.FileWriter(log_path)

        for i in range(num_env):

            if env_name == 'stochastic':
                cur_env = StochasticInputOffloadEnv(query_budget_frac=fraction)
            elif env_name == 'AQE':
                cur_env = AlwaysQueryEdgeOffloadEnv()
            elif env_name == 'FourAction':
                cur_env = FourActionOffloadEnv()
            else:
                pass

            cur_env._seed(seed + i)
            if algo == 'a2c':
                model = A2C(sess, n_s, n_a, total_step, i_thread=i, optimizer=optimizer,
                            lr=lr, model_config=parser['MODEL_CONFIG'], discrete=True)
            else:
                model = PPO(sess, n_s, n_a, total_step, i_thread=i, optimizer=optimizer,
                            lr=lr, clip=clip, model_config=parser['MODEL_CONFIG'], discrete=True)

            trainer = AsyncTrainer(cur_env, model, save_path, summary_writer, global_counter,
                                   i, lr_scheduler, beta_scheduler, model_summary, wt_summary,
                                   reward_summary=reward_summary, clip_scheduler=clip_scheduler)
            if i == 0:
                reward_summary = (trainer.reward_summary, trainer.total_reward)
            trainers.append(trainer)

    sess.run(tf.global_variables_initializer())
    global_model.init_train()
    saver = tf.train.Saver(max_to_keep=20)
    # global_model.load(saver, save_path)

    def train_fn(i_thread):
        trainers[i_thread].run(sess, saver, coord)

    for i in range(num_env):
        thread = threading.Thread(target=train_fn, args=(i,))
        thread.start()
        threads.append(thread)
    signal.signal(signal.SIGINT, signal_handler)
    signal.pause()
    coord.request_stop()
    coord.join(threads)
    save_flag = input('save final model? Y/N: ')
    if save_flag.lower().startswith('y'):
        print('saving model at step %d ...' % global_counter.cur_step)
        global_model.save(saver, save_path + 'checkpoint', global_counter.cur_step)

    # evaluation
    _, log_path = init_out_dir(base_dir, 'evaluate')
    evaluator = Evaluator(env, global_model, log_path, test_seeds)
    evaluator.run()
    data = env._output_result()
    if data is not None:
        data.to_csv(log_path + '/evaluate_result.csv')


def evaluate(parser, test_seeds, algo, env_name, base_dir):
    #fraction = parser.getfloat('ENV_CONFIG', 'QUARY_BUDGET_FRACTION')

    if env_name == 'stochastic':
        env = StochasticInputOffloadEnv(query_budget_frac=fraction)
    elif env_name == 'AQE':
        env = AlwaysQueryEdgeOffloadEnv()
    elif env_name == 'FourAction':
        env = FourActionOffloadEnv()
    else:
        pass

    n_a = env.n_a
    n_s = env.n_s
    sess = tf.Session()
    if algo == 'a2c':
        model = A2C(sess, n_s, n_a, -1, model_config=parser['MODEL_CONFIG'],
                    discrete=True)
    elif algo == 'ppo':
        model = PPO(sess, n_s, n_a, -1, model_config=parser['MODEL_CONFIG'],
                    discrete=True)
    else:
        model = None

    #base_dir = parser.get('TRAIN_CONFIG', 'BASE_DIR')
    save_path, log_path = init_out_dir(base_dir, 'evaluate')
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    model.load(saver, save_path)
    evaluator = Evaluator(env, model, log_path, test_seeds)
    evaluator.run()

    data = env._output_result()
    if data is not None:
        data.to_csv(log_path + '/evaluate_result.csv')


if __name__ == '__main__':
    args = parse_args()
    parser = configparser.ConfigParser()
    parser.read(args.config_path)
    test_seeds = [int(x) for x in args.test_seeds.split(',')]

    if args.train_seeds != 'False':
        train_seeds = [int(x) for x in args.train_seeds.split(',')]
    else:
        train_seeds = None

    print('main train seeds: ', train_seeds)
    print('ENV NAME: ', args.env_name, 'ALGO: ', args.agent)

    if args.mode == 'train':
        train(parser, train_seeds, test_seeds, args.agent, args.env_name, args.base_dir)
    elif args.mode == 'evaluate':
        evaluate(parser, test_seeds, args.agent, args.env_name, args.base_dir)

