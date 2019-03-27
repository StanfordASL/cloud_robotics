#!/usr/bin/env python3
import argparse
import configparser
import signal
import tensorflow as tf
import threading
import sys,os
import numpy as np

from agents.models import A2C, PPO
from train import Trainer, AsyncTrainer, Evaluator
from utils import GlobalCounter, init_out_dir, init_model_summary, signal_handler

CLOUD_ROOT_DIR = os.environ['CLOUD_ROOT_DIR']
#sys.path.append(CLOUD_ROOT_DIR + '/always_query_edge_simulator') 
#from always_query_edge_simulator import AlwaysQueryEdgeOffloadEnv
#from stochastic_simulator.stochastic_video_offload_env import StochasticInputOffloadEnv

sys.path.append(CLOUD_ROOT_DIR + '/simulate_RL/FaceNet_four_action_simulator') 
from four_action_simulator_v1_fnet import FourActionOffloadEnv

def parse_args():
    default_config_path = os.path.join(CLOUD_ROOT_DIR, 'rl_configs/test.ini')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, required=False,
                        default=default_config_path, help="config path")
    parser.add_argument('--agent', type=str, required=False,
                        default='a2c', help="a2c, ppo")
    parser.add_argument('--test-seeds', type=str, required=False,
                        default='100,200', help="seeds to test a pre-trained RL model")
    parser.add_argument('--query-budget-fraction-list', type=str, required=False,
                        default='100,200', help="query budget fraction list")
    parser.add_argument('--env-name', type=str, required=False,
                        default='stochastic', help="toggle between environments")
    parser.add_argument('--log-path', type=str, required=False, help="log path")
    parser.add_argument('--model-save-path', type=str, required=False, default='stochastic', help="where to load trained RL model")
    return parser.parse_args()


def RL_offload_evaluate(parser = None, test_seeds = None, algo = None, env_name = 'alwaysQueryEdge', query_budget_fraction_list = None, model_save_path = None, log_path = None):

    if env_name == 'stochastic':
        env = StochasticInputOffloadEnv()
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

    sess.run(tf.global_variables_initializer())
    
    saver = tf.train.Saver()
    model.load(saver, model_save_path)
    evaluator = RLOffloadEvaluator(env, model, log_path, test_seeds, query_budget_fraction_list)
    evaluator.run()

    # optional
    #data = env._output_result()
    #if data is not None:
    #    data.to_csv(log_path + '/evaluate_result.csv')


class RLOffloadEvaluator:
    def __init__(self, env, model, log_path, test_seeds, query_budget_fraction_list):
        self.env = env
        self.model = model
        self.algo = self.model.name
        self.log_path = log_path
        self.test_seeds = test_seeds
        self.data = []
        self.query_budget_fraction_list = query_budget_fraction_list

        self.test_cases = []
        for test_seed in self.test_seeds:
            for query_budget_fraction in self.query_budget_fraction_list:
                test_case = [test_seed, query_budget_fraction]
                self.test_cases.append(test_case)
        self.n_tests = len(self.test_cases)

    def perform(self, episode_i):
        self.env.controller_name = 'RL' 
        episode_seed = self.test_cases[episode_i][0] 
        episode_query_budget = self.test_cases[episode_i][1]
        print('#####################')
        print('start RL: seed', episode_seed, ', budget: ', episode_query_budget)
        self.env._seed(episode_seed)
        ob = self.env._reset(seed = episode_seed, fixed_query_budget = episode_query_budget)
        done = False
        rewards = []
        step = 0
        while True:
            _, policy = self.model.forward(ob, done, mode='p')
            action = np.argmax(policy)
            next_ob, reward, done, _ = self.env._step(action)
            cur_sample = {'episode': episode_i,
                          'step': step,
                          'state': ','.join(['%.2f' % x for x in ob]),
                          'policy': ','.join(['%.3f' % x for x in policy]),
                          'action': action,
                          'reward': reward}
            self.data.append(cur_sample)
            rewards.append(reward)
            if done:
                break
            ob = next_ob
            step += 1
        return np.sum(rewards)

    def run(self):
        total_rewards = []
        for i in range(self.n_tests):
            reward = self.perform(i)
            print('REWARD FOR TEST num', i, ' : ', reward)
            print('#####################')
            print(' ')
            total_rewards.append(reward)
        total_rewards = np.array(total_rewards)
        print('total reward mean: %.2f, std: %.2f' %
              (np.mean(total_rewards), np.std(total_rewards)))
        #df = pd.DataFrame(self.data)
        #df.to_csv(self.log_path + '/evaluation_mdp.csv')
        self.env.results_df.to_csv(self.log_path + '/RL_results_df.csv')


if __name__ == '__main__':
    args = parse_args()
    parser = configparser.ConfigParser()
    parser.read(args.config_path)
    test_seeds = [int(x) for x in args.test_seeds.split(',')]
    query_budget_fraction_list = [float(x) for x in args.query_budget_fraction_list.split(',')]
    model_save_path = args.model_save_path 
    log_path = args.log_path

    print('EVALUATING A TRAINED RL AGENT')
    print('ENV NAME: ', args.env_name)
    print('test_seeds: ', test_seeds)
    print('query_budget_fraction_list: ', query_budget_fraction_list)
    print('loading model from: ', model_save_path, ' logging: ', log_path)

    RL_offload_evaluate(parser = parser, test_seeds = test_seeds, algo = args.agent, env_name = args.env_name, query_budget_fraction_list = query_budget_fraction_list, model_save_path = model_save_path, log_path = log_path)
