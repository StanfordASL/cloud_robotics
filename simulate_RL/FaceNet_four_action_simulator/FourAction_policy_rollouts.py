# run benchmark controllers on a series of traces for the offloading problem
# save their results in a csv for later analysis

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
import sys, os
import numpy as np
import itertools
import gym
from gym import spaces
from gym.utils import seeding
import pandas

CLOUD_ROOT_DIR=os.environ['CLOUD_ROOT_DIR']
sys.path.append(CLOUD_ROOT_DIR)

UTILS_DIR = CLOUD_ROOT_DIR + '/utils/'
sys.path.append(UTILS_DIR)

from textfile_utils import flatten_list
from plotting_utils import *
from four_action_simulator_v1_fnet import FourActionOffloadEnv
from v1_control_policies_FourAction import * 
from calculation_utils import *
from RSS_data_parsing_utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, required=False,
                        default='v1Sim')
    parser.add_argument('--test-seeds', type=str, required=False, default="10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200")
    parser.add_argument('--query-budget-fraction-list', type=str, required=False, default="0.10,0.20,0.50,0.70,1.0")
    parser.add_argument('--base-results-dir', type=str, required=True, default=None)
    return parser.parse_args()

if __name__ == '__main__':

    DATA_DELETE_MODE = True

    args = parse_args()
    prefix = args.prefix

    base_results_dir = args.base_results_dir

    base_data_dir = base_results_dir + '/FourAction_FaceNet_baseline_data_' + prefix + '/'

    if DATA_DELETE_MODE:
        remove_and_create_dir(base_data_dir)

    query_budget_frac_list = [float(x) for x in args.query_budget_fraction_list.split(',')]
    seed_list = [int(x) for x in args.test_seeds.split(',')]

    offloader_env = FourActionOffloadEnv() 
    offloader_env.print_mode = False

    #threshold_list = [0.05, 0.10, 0.15, 0.25, 0.5]
    threshold_list = [0.5]

    for fixed_query_budget in query_budget_frac_list: 
        shuffle_mode = True
        
        for seed in seed_list:
            # all edge
            FourAction_rollout_all_edge(offloader_env = offloader_env, shuffle_mode = shuffle_mode, seed = seed, fixed_query_budget = fixed_query_budget)

            # all cloud
            FourAction_rollout_all_cloud(offloader_env = offloader_env, shuffle_mode = shuffle_mode, seed = seed, fixed_query_budget = fixed_query_budget)

            # now implement a random action
            FourAction_rollout_random_action(offloader_env = offloader_env, shuffle_mode = shuffle_mode, seed = seed, fixed_query_budget = fixed_query_budget)

            ### now implement oracle action
            for threshold in threshold_list:
                FourAction_rollout_threshold(offloader_env = offloader_env, shuffle_mode = shuffle_mode, seed = seed, results_print_mode = True, fixed_query_budget = fixed_query_budget, threshold = threshold)

            ### now implement oracle action
            reward_vec, empirical_reward_vec, control_action_vec, implemented_action_vec, time_vec = FourAction_rollout_pure_oracle_action(offloader_env = offloader_env, shuffle_mode = shuffle_mode, seed = seed, results_print_mode = True, GP_mode = False, fixed_query_budget = fixed_query_budget)
  
            #print(reward_vec)
            #print(empirical_reward_vec)
            #print(control_action_vec)
            #print(implemented_action_vec)
            #print(time_vec)

    # ROLLOUTS ARE DONE, now collate results
    ###############################################
    # mean of results grouped by controller name
    # results_df.groupby('controller_name').mean()
    results_df = offloader_env.results_df
    
    results_csv = base_data_dir + '/FourAction_episode_results.csv'
    
    results_df.to_csv(results_csv)

    controller_results_csv = base_data_dir + '/FourAction_episode_summary_results.csv'

    controller_rewards_df = results_df_to_controller_rewards_df(results_df = results_df, controller_results_csv_fname = controller_results_csv)

    # print four action controller rewards df summary
    ###############################################
    print('SUMMARY REWARDS STATS')
    print(controller_rewards_df.groupby('controller_name')['reward_sum'].median())
