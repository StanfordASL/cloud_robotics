import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

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

from textfile_utils import *
from state_utils_four_action_simulator import *
from calculation_utils import *


# CHANGED FOR THE V1 SIMULATOR !!!!
# DONE
def FourAction_rollout_all_edge(offloader_env = None, shuffle_mode = None, seed = None, old_edge_action = 0,  curr_edge_action = 2, results_print_mode = True, fixed_query_budget = None):

    # start the episode
    offloader_env._reset(shuffle_mode = shuffle_mode, seed = seed, fixed_query_budget = fixed_query_budget)

    offloader_env.controller_name = 'curr_edge'

    done_flag = False

    reward_vec = []
    while (not done_flag):
        
	# decision stage capping handled by simulator
        new_state_dict, reward, done_flag, _ = offloader_env._step(curr_edge_action)

        reward_vec.append(reward)

    if results_print_mode:
        print('#######################')
        print('END EPISODE FourActionV1 all-edge: ')
        print('done action: ', old_edge_action, curr_edge_action, ' , flag : ', done_flag)
        print('time: ', offloader_env.t)
        print('mean reward: ', round(np.mean(reward_vec), 3))
        print('#######################')
        print(' ')


# DONE
def FourAction_rollout_threshold(offloader_env = None, shuffle_mode = None, seed = None, old_edge_action = 0,  curr_edge_action = 2, curr_cloud_action = 3, results_print_mode = True, fixed_query_budget = None, threshold = 0.5):

    # start the episode
    offloader_env._reset(shuffle_mode = shuffle_mode, seed = seed, fixed_query_budget = fixed_query_budget)

    offloader_env.controller_name = 'threshold-' + str(threshold)

    done_flag = False

    reward_vec = []

    next_planned_action = curr_edge_action

    while (not done_flag):
        
	# decision stage capping handled by simulator
        new_state_dict, reward, done_flag, _ = offloader_env._step(next_planned_action)

	conf = offloader_env.state_dict['past_overall_predict'][1]

	# edge conf too low
	if float(conf) < threshold:	
	    next_planned_action = curr_cloud_action
            #print('conf: ', conf, 'threshold: ', threshold, next_planned_action)
	else:
	    next_planned_action = curr_edge_action

        reward_vec.append(reward)

    if results_print_mode:
        print('#######################')
        print('END EPISODE FourActionV1 THRESHOLD: ', str(threshold))
        print('done action: ', old_edge_action, curr_edge_action, ' , flag : ', done_flag)
        print('time: ', offloader_env.t)
        print('mean reward: ', round(np.mean(reward_vec), 3))
        print('#######################')
        print(' ')


def FourAction_rollout_all_cloud(offloader_env = None, shuffle_mode = None, seed = None, past_cloud_action = 1, curr_cloud_action = 3, results_print_mode = True, fixed_query_budget = None):

    # start the episode
    offloader_env._reset(shuffle_mode = shuffle_mode, seed = seed, fixed_query_budget = fixed_query_budget)

    # decide interval at which to query the cloud
    T = offloader_env.T
    max_num_queries = offloader_env.max_num_queries

    # cloud has the query budget fraction 
    offloader_env.controller_name = 'curr_cloud'

    cloud_query_interval = int(T/max_num_queries)
    done_flag = False
    reward_vec = []

    while (not done_flag):
        
        # have a budget to query cloud and at the right polling interval
        if (offloader_env.t % cloud_query_interval == 0):
            new_state_dict, reward, done_flag, _ = offloader_env._step(curr_cloud_action)
        else:
            new_state_dict, reward, done_flag, _ = offloader_env._step(past_cloud_action)

        reward_vec.append(reward)

    if results_print_mode:
        print('#######################')
        print('END EPISODE FourActionV1 all-cloud')
        print('done action: ', past_cloud_action, curr_cloud_action, ' , flag : ', done_flag)
        print('time: ', offloader_env.t)
        print('mean reward: ', round(np.mean(reward_vec), 3))
        print('#######################')
        print(' ')


# DONE
# random action
def FourAction_rollout_random_action(offloader_env = None, shuffle_mode = None, seed = None, results_print_mode = True, fixed_query_budget = None):
    # start the episode
    offloader_env._reset(shuffle_mode = shuffle_mode, seed = seed, fixed_query_budget = fixed_query_budget)

    offloader_env.controller_name = 'random'
    action_list = range(len(offloader_env.numeric_to_action_dict))

    done_flag = False
    reward_vec = []
    while (not done_flag):
	random_action = random.sample(action_list, 1)[0]
	new_state_dict, reward, done_flag, _ = offloader_env._step(random_action)
        reward_vec.append(reward)

    if results_print_mode:
        print('#######################')
        print('END EPISODE FourActionV1 random')
        print('time: ', offloader_env.t)
        print('mean reward: ', round(np.mean(reward_vec), 3))
        print(' ')


# NOT DONE, HAVING TROUBLE WITH THE LOGIC HERE!
# oracle action, but NO constraints!
def FourAction_rollout_pure_oracle_action(offloader_env = None, shuffle_mode = None, seed = None, results_print_mode = False, GP_mode = True, fixed_query_budget = None, past_edge_action = 0, past_cloud_action = 1, curr_edge_action = 2, curr_cloud_action = 3):

    # start the episode
    offloader_env._reset(shuffle_mode = shuffle_mode, seed = seed, fixed_query_budget = fixed_query_budget)
    offloader_env.controller_name = 'pure_oracle'
    
    done_flag = False
    reward_vec = []

    control_action_vec, sparse_control_action_vec, edge_reward_vec, cloud_reward_vec, sparse_gap_vec, empirical_reward_vec = offline_oracle_strategy(offloader_env = offloader_env, shuffle_mode = shuffle_mode, seed = seed, fixed_query_budget = fixed_query_budget)

    # now implement this control policy online
    done_flag = False
    reward_vec = []
    implemented_action_vec = []

    past_action = None
    time_vec = []

    while (not done_flag):
        
        oracle_action = control_action_vec[offloader_env.t]
        
        if past_action == curr_edge_action:
            oracle_action = past_edge_action
        if past_action == curr_cloud_action:
            oracle_action = past_cloud_action
            
        past_action = oracle_action
        time_vec.append(offloader_env.t)
	
        new_state_dict, reward, done_flag, _ = offloader_env._step(oracle_action)
        reward_vec.append(reward)
        implemented_action_vec.append(oracle_action)

    if results_print_mode:
        print('#######################')
        print('END EPISODE FourActionV1 NEW ORACLE')
        print('time: ', offloader_env.t)
        print('mean reward: ', round(np.mean(reward_vec), 3), 'empirical reward: ', round(np.mean(empirical_reward_vec), 3))
        print(' ')

    return reward_vec, empirical_reward_vec, control_action_vec, implemented_action_vec, time_vec

def offline_oracle_strategy(offloader_env = None, shuffle_mode = None, seed = None, fixed_query_budget = None, past_edge_action = 0, past_cloud_action = 1, curr_edge_action = 2, curr_cloud_action = 3):

    # start the episode
    offloader_env._reset(shuffle_mode = shuffle_mode, seed = seed, fixed_query_budget = fixed_query_budget)
    
    done_flag = False
    reward_vec = []
   
    # query ts and edge cloud accuracy gap
    query_ts = offloader_env.query_ts
    edge_cloud_accuracy_gap_vec = offloader_env.edge_cloud_accuracy_gap_vec
    edge_prediction_vec = offloader_env.edge_prediction_vec
    edge_confidence_vec = offloader_env.edge_confidence_vec
    cloud_prediction_vec = offloader_env.cloud_prediction_vec
    true_value_vec = offloader_env.true_value_vec

    # compute the reward in all these cases
    past_value = None
    changepoints_vec = []

    control_action_vec = []
    sparse_control_action_vec = []
    edge_reward_vec = []
    cloud_reward_vec = []
    sparse_gap_vec = []
    reward_vec = []

    for t, gap in enumerate(query_ts):
        # true value
        y_true = true_value_vec[t]
        
        if gap != past_value:
            past_value = gap
            changepoints_vec.append(t)
            sparse_gap_vec.append(gap)        


            # compute the edge reward
            edge_state_dict = {}
            edge_state_dict['curr_query_x'] = [query_ts[t]]

            past_edge_overall_predict = [edge_prediction_vec[t], edge_confidence_vec[t]]
            edge_state_dict['past_overall_predict'] = past_edge_overall_predict

            # what would the edge cost be?
            edge_reward, edge_query_cost_term, edge_accuracy_cost_term, _ = get_FourAction_reward(state_dict = edge_state_dict, numeric_action = curr_edge_action, reward_params_dict = offloader_env.reward_params_dict, numeric_to_action_dict = offloader_env.numeric_to_action_dict, query_cost_dict = offloader_env.query_cost_dict, GP_mode = False, y_true_input = y_true) 
            
            edge_reward_vec.append(edge_reward)

            # compute the cloud reward
            cloud_state_dict = {}
            cloud_state_dict['curr_query_x'] = [query_ts[t]]
            past_cloud_overall_predict = [cloud_prediction_vec[t], 1.0]
            cloud_state_dict['past_overall_predict'] = past_cloud_overall_predict

            # what would the cloud cost be?
            cloud_reward, cloud_query_cost_term, cloud_accuracy_cost_term, _ = get_FourAction_reward(state_dict = cloud_state_dict, numeric_action = curr_cloud_action, reward_params_dict = offloader_env.reward_params_dict, numeric_to_action_dict = offloader_env.numeric_to_action_dict, query_cost_dict = offloader_env.query_cost_dict, GP_mode = False, y_true_input = y_true) 

            cloud_reward_vec.append(cloud_reward)

            if edge_reward >= cloud_reward:
                optimal_action = curr_edge_action
                reward = edge_reward
            else:
                optimal_action = curr_cloud_action
                reward = cloud_reward

            past_optimal_action = optimal_action
            sparse_control_action_vec.append(optimal_action)
            control_action_vec.append(optimal_action) 
            reward_vec.append(reward)

        else:
            if past_optimal_action == curr_edge_action:
                action = past_edge_action
                past_overall_predict = past_edge_overall_predict
            else:
                action = past_cloud_action
                past_overall_predict = past_cloud_overall_predict
            control_action_vec.append(action)

            test_state_dict = {}
            test_state_dict['curr_query_x'] = [query_ts[t]]
            test_state_dict['past_overall_predict'] = past_overall_predict

            reward, query_cost_term, accuracy_cost_term, _ = get_FourAction_reward(state_dict = test_state_dict, numeric_action = action, reward_params_dict = offloader_env.reward_params_dict, numeric_to_action_dict = offloader_env.numeric_to_action_dict, query_cost_dict = offloader_env.query_cost_dict, GP_mode = False, y_true_input = y_true) 
            
            reward_vec.append(reward)

    return  control_action_vec, sparse_control_action_vec, edge_reward_vec, cloud_reward_vec, sparse_gap_vec, reward_vec

