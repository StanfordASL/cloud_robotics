import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import sys, os
import numpy as np
import itertools
import copy
from random import shuffle
import pandas
import random

CLOUD_ROOT_DIR=os.environ['CLOUD_ROOT_DIR']
sys.path.append(CLOUD_ROOT_DIR)
UTILS_DIR = CLOUD_ROOT_DIR + '/utils/'
sys.path.append(UTILS_DIR)

from textfile_utils import *
from calculation_utils import *

MAX_L2_NORM=2.0
MIN_Y = 0.0
MAX_Y = 11.0

"""
"""
def get_FourAction_initial_state_v1(curr_query_x=0, past_edge_predict=0, past_edge_conf=0,
                                    past_cloud_predict=0, num_queries_remain=0, num_past_states=1):
    state_dict = {}
    state_dict['curr_query_x'] = [curr_query_x]
    state_dict['curr_xdiff'] = [0.]
    state_dict['past_edge_predict'] = [past_edge_predict]
    #state_dict['past_edge_predict_vector'] = [past_edge_predict for _ in range(num_past_states)]
    state_dict['past_edge_conf'] = [past_edge_conf]
    #state_dict['past_edge_conf_vector'] = [past_edge_conf for _ in range(num_past_states)]
    state_dict['past_edge_x'] = [curr_query_x]
    #state_dict['past_edge_x_vector'] = [curr_query_x for _ in range(num_past_states)]
    state_dict['past_edge_xdiff'] = [0.]
    state_dict['past_edge_tdiff'] = [0.]
    #state_dict['past_edge_query_time'] = [0.]

    state_dict['past_cloud_predict'] = [past_cloud_predict]
    #state_dict['past_cloud_predict_vector'] = [past_cloud_predict for _ in range(num_past_states)]
    state_dict['past_cloud_x'] = [curr_query_x]
    #state_dict['past_cloud_x_vector'] = [curr_query_x for _ in range(num_past_states)]
    state_dict['past_cloud_xdiff'] = [0.]
    state_dict['past_cloud_tdiff'] = [0.]
    #state_dict['past_cloud_query_time'] = [0.0]
    
    state_dict['past_overall_predict'] = [MIN_Y, 0.0] # for reward computation only
    state_dict['num_queries_remain'] = [num_queries_remain]
    state_dict['edge_queried'] = [1]
    state_dict['cloud_queried'] = [1]
    return state_dict

def get_FourAction_min_state_v1(MULT=0.5):
    
    state_dict = {}
    #state_dict['curr_query_x'] = [-abs_x]
    state_dict['curr_xdiff'] = [0.]
    state_dict['past_edge_predict'] = [MIN_Y]
    state_dict['past_edge_conf'] = [0.]
    #state_dict['past_edge_x'] = [-abs_x]
    # TODO: get better normalization for xdiff
    #state_dict['past_edge_xdiff'] = [-MAX_L2_NORM*MULT]
    state_dict['past_edge_xdiff'] = [0.]
    state_dict['past_edge_tdiff'] = [0.]

    state_dict['past_cloud_predict'] = [MIN_Y]
    #state_dict['past_cloud_x'] = [-abs_x]
    #state_dict['past_cloud_xdiff'] = [-MAX_L2_NORM*MULT]
    state_dict['past_cloud_xdiff'] = [0.]
    state_dict['past_cloud_tdiff'] = [0.]
    state_dict['past_overall_predict'] = [MIN_Y, 0.]

    state_dict['num_queries_remain'] = [0]
    state_dict['edge_queried'] = [0]
    state_dict['cloud_queried'] = [0]
    return state_dict

def get_FourAction_max_state_v1(T=200, max_num_queries=200, MULT=0.5):
    
    state_dict = {}
    #state_dict['curr_query_x'] = [abs_x]
    state_dict['curr_xdiff'] = [MAX_L2_NORM]
    state_dict['past_edge_predict'] = [MAX_Y]
    state_dict['past_edge_conf'] = [1.]
    #state_dict['past_edge_x'] = [abs_x]
    state_dict['past_edge_xdiff'] = [MAX_L2_NORM]
    state_dict['past_edge_tdiff'] = [T*MULT]

    state_dict['past_cloud_predict'] = [MAX_Y]
    #state_dict['past_cloud_x'] = [abs_x]
    state_dict['past_cloud_xdiff'] = [MAX_L2_NORM]
    state_dict['past_cloud_tdiff'] = [T*MULT]
    state_dict['past_overall_predict'] = [MAX_Y, 1.0]
    
    state_dict['num_queries_remain'] = [max_num_queries]
    state_dict['edge_queried'] = [1]
    state_dict['cloud_queried'] = [1]
    return state_dict

def get_FourAction_state_bounds_v1(T, max_num_queries):
    min_state_dict = get_FourAction_min_state_v1()
    max_state_dict = get_FourAction_max_state_v1(T=T, max_num_queries=max_num_queries)
    return min_state_dict, max_state_dict

def FourAction_report_rewards(
                    results_df = None,
                    reward = None,
                    episode_number=None,
                    t=None,
                    control_action=None,
                    query_cost_term = None,
                    accuracy_cost_term = None,
                    accuracy_weight = None,
                    query_cost_weight = None,
                    controller_name = None,
                    y_pred = None,
                    y_pred_conf = None,
                    y_true = None,
                    cloud_query_budget_frac = None,
                    edge_predict_y = None, 
                    edge_confidence = None, 
                    cloud_predict_y = None, 
                    cloud_confidence = None, 
                    edge_cloud_accuracy_gap = None, 
                    input_query_x = None,
                    seen_value = None, 
                    action_to_numeric_dict = None, 
                    query_cost_dict = None, 
                    rolling_diff_query_x = None,
                    decision_stage = None):

    numeric_control_action = action_to_numeric_dict[control_action]

    # dictionary of useful data 
    df_dict = {}
    df_dict['reward'] = [reward]
    df_dict['episode_number'] = [episode_number]
    df_dict['t'] = [t]
    df_dict['control_action'] =  [control_action]
    df_dict['numeric_control_action'] =  [numeric_control_action]

    # new ones
    df_dict['query_cost_term'] =  [query_cost_term]
    df_dict['accuracy_cost_term'] =  [accuracy_cost_term]
    df_dict['accuracy_weight'] =  [accuracy_weight]
    df_dict['query_cost_weight'] =  [query_cost_weight]
    df_dict['controller_name'] =  [controller_name]
    df_dict['y_pred'] =  [y_pred]
    df_dict['y_true'] =  [y_true]
    df_dict['cloud_query_budget_frac'] =  [cloud_query_budget_frac]

    df_dict['edge_predict_y'] = [edge_predict_y]
    df_dict['edge_confidence'] = [edge_confidence]
    df_dict['cloud_predict_y'] = [cloud_predict_y]
    df_dict['cloud_confidence'] = [cloud_confidence]

    df_dict['edge_cloud_accuracy_gap'] = [edge_cloud_accuracy_gap]
    df_dict['input_query_x'] = [input_query_x]
    df_dict['seen_value'] = [seen_value]
    df_dict['rolling_diff_query_x'] = [rolling_diff_query_x]
    df_dict['decision_stage'] = [decision_stage]

    # convert to df
    local_df = pandas.DataFrame(df_dict)
    # append to existing dataframe of previous time results
    results_df = results_df.append(local_df)
    return results_df

"""
done, technically same as before
"""

def get_FourAction_reward(state_dict = None, numeric_action = None, reward_params_dict = None, numeric_to_action_dict = None, query_cost_dict = None, print_mode = False, y_true_input = None, noise_level = 0.05, PROBE_COST_ONLY = False):
   
    # weights for accuracy, probes
    query_cost_weight = reward_params_dict['query_cost_weight']
    accuracy_weight = reward_params_dict['accuracy_weight']

    prediction_confidence = state_dict['past_overall_predict']
    prediction = prediction_confidence[0]
    confidence = prediction_confidence[1]

    # add some noise to this
    query_noise = 0.0
    #query_noise = np.random.normal(0.0, noise_level)

    query_cost_term = query_cost_dict[numeric_action] + query_noise

    # only penalize the cost of probes
    if PROBE_COST_ONLY:
        reward = - query_cost_weight * query_cost_term
        accuracy_cost_term = 0.0
    # also penalize accuracy, full reward
    else:
        # accuracy depends on current prediction, which is classification error here
        if prediction == y_true_input:
            accuracy_cost_term = 0.0
        else:
            accuracy_cost_term = 1.0

        reward = - query_cost_weight * query_cost_term - accuracy_weight * accuracy_cost_term

    if print_mode:
        print(' ')
        print('reward: ', reward)
        print('query_cost_term: ', query_cost_term)
        print('accuracy_cost_term: ', accuracy_cost_term)
        print(' ')

    return reward, query_cost_term, accuracy_cost_term, y_true_input
