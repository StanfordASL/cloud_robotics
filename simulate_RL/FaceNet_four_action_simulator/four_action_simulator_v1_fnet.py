import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import sys, os
import numpy as np
import itertools
import gym
from gym import spaces
from gym.utils import seeding
import pandas
try:
    import ConfigParser
except:
    import configparser

CLOUD_ROOT_DIR=os.environ['CLOUD_ROOT_DIR']
sys.path.append(CLOUD_ROOT_DIR)

UTILS_DIR = CLOUD_ROOT_DIR + '/utils/'
sys.path.append(UTILS_DIR)

from state_utils_four_action_simulator_fnet import *
from calculation_utils import *

FACENET_UTILS_DIR = CLOUD_ROOT_DIR + '/simulate_RL/facenet/'
sys.path.append(FACENET_UTILS_DIR)

# generates a synthetic video stream of faces
from stochastic_timeseries_facenet import facenet_stochastic_video

# pre-computed facenet images to aid in simulation training
DATA_DIR = CLOUD_ROOT_DIR + '/data/data_facenet/'
FACENET_DATA_CSV = DATA_DIR + '/SVM_results.csv'


"""
    uses real faces from FaceNet
    environment (MDP) for mobile offloading

    - states: 
        - past predictions and current query
        - optionally: limited budget
    - actions:
        - 0: past edge prediction [cost 0]
        - 1: past cloud [cost 0]
        - 2: new edge [medium cost, 0.3]
        - 3: new cloud [high cost, 1.0]
    - reward:
        - weighted sum of accuracy and query cost
        - optionally: huge penalty if we go above the budget
    - dynamics: past predictions are updated as we make decisions
"""


FOUR_ACTION_ORDER_LIST = ['curr_xdiff',
                          'past_edge_predict',
                          'past_edge_conf',
                          'past_edge_xdiff',
                          'past_edge_tdiff',
                          'past_cloud_predict',
                          'past_cloud_xdiff',
                          'past_cloud_tdiff',
                          'past_overall_predict',
                          'num_queries_remain',
                          'edge_queried',
                          'cloud_queried']


class FourActionOffloadEnv:

    """ Init the environment
    """
    def __init__(self, file_path_config_file=None, base_results_dir = None, controller_name = None, query_budget_frac = 1.0, num_past_states = 3, facenet_data_csv = FACENET_DATA_CSV):

        # where to store results
        ######################################
        self.base_results_dir = base_results_dir
        self.controller_name = controller_name
        self.print_mode = False
        self.SVM_results_df = pandas.read_csv(facenet_data_csv)

        # reward params
        # eventually get this from a file
        ######################################
        self.reward_params_dict = {}
        self.reward_params_dict['query_cost_weight'] = 1.0
        self.reward_params_dict['accuracy_weight'] = 10.0
        self.exceed_query_cost_budget = 0.0
        self.reward_params_dict['exceed_query_cost_budget'] = self.exceed_query_cost_budget

        # costs of basic queries
        self.query_cost_dict = {}
        self.query_cost_dict[0] = 0.0
        self.query_cost_dict[1] = 0.0
        self.query_cost_dict[2] = 1.0
        self.query_cost_dict[3] = 5.0

        # budget on queries randomly changed in __RESET__ now!!
        self.seed = 42

        # action space
        ######################################
        self.action_to_numeric_dict = {}
        self.action_to_numeric_dict['past_edge'] = 0
        self.action_to_numeric_dict['past_cloud'] = 1
        self.action_to_numeric_dict['curr_edge'] = 2
        self.action_to_numeric_dict['curr_cloud'] = 3

        # discrete action space
        self.numeric_to_action_dict = ['past_edge', 'past_cloud', 'curr_edge', 'curr_cloud']
        self.n_a = len(self.numeric_to_action_dict)
        print('num_actions: ', self.n_a)

        # state vector bounds
        ######################################
        self.num_past_states = num_past_states

        # state space
        # HACK
        self.n_s = len(FOUR_ACTION_ORDER_LIST) + 1
        #self.n_s = len(FOUR_ACTION_ORDER_LIST)

        #print('MIN STATE LEN: ', self.n_s)
        #print('MAX STATE LEN: ', len(self.max_state_vector))

        # query ts
        self.GP_mode = False

        # LOGGING
        # a dataframe of state, action, reward history for logging
        ######################################
        self.results_df = pandas.DataFrame()
        self.t = 0
        self.episode_number = 0

        self.TRAIN_MODE = False
        self.LOG_MODE = True
        self.CLOUD_CONF = 1.0
 
        if (self.print_mode):
            print('FUNCTION init')
            print('action space', self.action_space)
            print('observation space', self.observation_space)

    """ 
        Dynamics: given s,a return s', reward, done
    """
    def _step(self, action):
        #print('action: ', action)

        nominal_action_name = self.numeric_to_action_dict[action]
        action_name = self._get_action_name(nominal_action_name)
        # reward compuation is based on capped action
        final_action = self.action_to_numeric_dict[action_name]
        stage_end = self._simulate_action_updates(action_name)

        self.stage_end = stage_end

        if not stage_end:
            # now only penalize for the probe, hold off on penalizing accuracy
            reward, y_true = self._compute_reward(final_action, stage_end)

        # case where we are on the second part of the decision stage
        # update the time and new queries here
        else:

            # how to shift the state is the same as before
            # BUT, we now give the final reward which incorporates the accuracy loss only at END
            # ALSO has weighted cost of probes for BOTH actions

            # get the reward based on the true value
            y_true_input = self.true_value_vec[self.t]

            # now evaluate the full reward!
            reward, y_true = self._compute_reward(final_action, stage_end, y_true_input=y_true_input)

            # only advance the time and query at the end of the second stage
            self.t += 1

            # now update the decision stage
            self.edge_queried = 0
            self.cloud_queried = 0

            # otherwise we have reached the end and we trigger the done_flag loop
            if self.t < len(self.query_ts):
                # update state, update sensory information

                # UPDATE FOR FNET, no cur_query here
                cur_x = self.query_ts[self.t]
                self.state_dict['curr_query_x'] = [cur_x]

                if self.t == 0:
                    curr_xdiff = 0.0
                else:
                    curr_xdiff = distance(cur_x, self.query_ts[self.t-1])

                self.state_dict['curr_xdiff'] = [curr_xdiff]
                self.state_dict['edge_queried'] = [self.edge_queried]
                self.state_dict['cloud_queried'] = [self.cloud_queried]
                past_x = self.state_dict['past_edge_x'][0]
                #self.state_dict['past_edge_xdiff'] = [cur_x - past_x]
                self.state_dict['past_edge_xdiff'] = [distance(cur_x, past_x)]
                self.state_dict['past_edge_tdiff'] = [self.state_dict['past_edge_tdiff'][0] + 1]
                past_x = self.state_dict['past_cloud_x'][0]
                #self.state_dict['past_cloud_xdiff'] = [cur_x - past_x]
                self.state_dict['past_cloud_xdiff'] = [distance(cur_x, past_x)]
                self.state_dict['past_cloud_tdiff'] = [self.state_dict['past_cloud_tdiff'][0] + 1]

        # always add in reward
        self.episode_reward_vec.append(reward)
        #self.action_chosen_vec[action] += 1.0
        self.action_chosen_vec[final_action] += 1.0

        # if done with episode
        if self.t == len(self.query_ts):
            done_flag = True
            if self.print_mode:
                print(np.sum(self.episode_reward_vec))
            print('FourActionSimulator-FACENET seed', self.seed, 'Controller: ', self.controller_name)
            print('episode mean/median reward: ', round(np.mean(self.episode_reward_vec),3),  round(np.median(self.episode_reward_vec), 3))
            print('action_diversity: ', self.action_chosen_vec)
            print('num queries remain: ', self.num_queries_remain)
            print('random query budget frac: ', self.query_budget_frac)
            print(' ')

        else:
            done_flag = False
            if self.LOG_MODE:
                rolling_diff_query_x = distance(self.query_ts[self.t], self.query_ts[self.t-1])

                self.results_df = FourAction_report_rewards(results_df=self.results_df,
                                                            reward=reward, 
                                                            episode_number=self.episode_number,
                                                            t=self.t,
                                                            control_action=action_name,
                                                            query_cost_term=self.query_cost_term,
                                                            accuracy_cost_term=self.accuracy_cost_term,
                                                            accuracy_weight=self.reward_params_dict['accuracy_weight'],
                                                            query_cost_weight=self.reward_params_dict['query_cost_weight'],
                                                            controller_name=self.controller_name,
                                                            y_pred=self.state_dict['past_overall_predict'][0],
                                                            y_pred_conf=self.state_dict['past_overall_predict'][1],
                                                            y_true=y_true,
                                                            cloud_query_budget_frac=self.query_budget_frac,
                                                            edge_predict_y=self.edge_prediction_vec[self.t],
                                                            edge_confidence=self.edge_confidence_vec[self.t],
                                                            cloud_predict_y=self.cloud_prediction_vec[self.t],
                                                            cloud_confidence=1.0,
                                                            edge_cloud_accuracy_gap=self.edge_cloud_accuracy_gap_vec[self.t],
                                                            input_query_x=self.query_ts[self.t],
                                                            seen_value=self.seen_vec[self.t],
                                                            action_to_numeric_dict=self.action_to_numeric_dict,
                                                            query_cost_dict=self.query_cost_dict,
                                                            rolling_diff_query_x=rolling_diff_query_x,
                                                            decision_stage=stage_end)


        #print('DEBUG: ', new_state_dict)
        new_state = AQE_state_dict_to_state_vec(order_list=FOUR_ACTION_ORDER_LIST, state_dict=self.state_dict)
        #print('NEW STATE LEN: ', len(new_state), new_state)
        #print('normed step state: ', self._norm_state(new_state))

        return self._norm_state(new_state), reward, done_flag, {}

    """
        FRONTIER
        reset initial state
    """
    # reset initial state
    def _reset(self, shuffle_mode = False, overwrite_results_df = False, seed = None, fixed_query_budget = None, coherence_time = 8, P_SEEN = 0.99, train_test = 'TRAIN', T = 80):

        if overwrite_results_df: 
            self.results_df = pandas.DataFrame()

        # reset initial time
        self.t = 0

        if seed is None:
            seed = self.seed
        else:
            self.seed = seed

        print('FACENET 4 action reset seed: ', seed)
        # sample a new timeseries
        self.edge_confidence_vec, self.edge_prediction_vec, self.cloud_prediction_vec, self.true_value_vec, self.edge_cloud_accuracy_gap_vec, self.query_ts, self.seen_vec, self.rolling_diff_vec, self.image_name_vec, self.train_test_membership_vec, self.embedding_norm_vec = facenet_stochastic_video(SVM_results_df = self.SVM_results_df, seed = seed, T = T, coherence_time = coherence_time, P_SEEN = 0.6, train_test_membership = train_test)

        # now choose uniform query fraction budget
        if fixed_query_budget:
            self.query_budget_frac = fixed_query_budget
            print('fixed query budget:', self.query_budget_frac)
        else:
            self.query_budget_frac_choices = [0.10, 0.20, 0.50, 0.70, 1.0]
            #self.query_budget_frac_choices = [0.50]
            self.query_budget_frac = np.random.choice(self.query_budget_frac_choices)

        self.T = len(self.query_ts) 

        self.max_num_queries = int(len(self.query_ts) * self.query_budget_frac)
        self.num_queries_remain = self.max_num_queries

        # get per-episode normalization factor
        min_state_dict, max_state_dict = get_FourAction_state_bounds_v1(self.T, self.max_num_queries)
        self.min_state_vector = AQE_state_dict_to_state_vec(order_list=FOUR_ACTION_ORDER_LIST, state_dict=min_state_dict)
        self.max_state_vector = AQE_state_dict_to_state_vec(order_list=FOUR_ACTION_ORDER_LIST, state_dict=max_state_dict)

        # initial state
        self.state_dict = get_FourAction_initial_state_v1(curr_query_x=self.query_ts[self.t],
                                                          past_edge_predict=self.edge_prediction_vec[self.t],
                                                          past_edge_conf=self.edge_confidence_vec[self.t],
                                                          past_cloud_predict=self.cloud_prediction_vec[self.t],
                                                          num_queries_remain=self.num_queries_remain,
                                                          num_past_states=self.num_past_states)
        self.edge_queried = self.state_dict['edge_queried'][0]
        self.cloud_queried = self.state_dict['cloud_queried'][0]
        #print('INIT STATE LEN: ', len(self.min_state_vector), len(self.max_state_vector))
        #print('MIN: ', self.min_state_vector)
        #print('MAX: ', self.max_state_vector)


        # results df
        self.episode_number += 1
        self.episode_reward_vec = []
        self.action_chosen_vec = [0.0 for a in range(self.n_a)]

        state = AQE_state_dict_to_state_vec(order_list=FOUR_ACTION_ORDER_LIST, state_dict=self.state_dict)
        #print('reset', len(state))
        #print('normed reset state: ', self._norm_state(state))
        return self._norm_state(state)

    def _seed(self, seed):
        np.random.seed(seed)
        self.seed = seed

    def _output_result(self):
        return None

    def _norm_state(self, state):
        norm_state = (state - self.min_state_vector) / (self.max_state_vector - self.min_state_vector)
        norm_state = np.clip(norm_state, 0, 1)
        #for state_name in ['past_edge_xdiff', 'past_cloud_xdiff',
        #                   'curr_query_x', 'curr_xdiff', 'past_edge_x', 'past_cloud_x',
        #                   'past_edge_predict', 'past_cloud_predict']:
        for state_name in ['past_edge_predict', 'past_cloud_predict']:
            i = FOUR_ACTION_ORDER_LIST.index(state_name)
            norm_state[i] = 2 * (norm_state[i]-0.5)
        return norm_state

    def _get_action_name(self, nominal_action_name):
        # if we want to override cloud queries above the budget and 
        # default them to an edge query
        action_name = nominal_action_name
        if self.num_queries_remain <= 0:
            if action_name == 'curr_cloud':
                action_name = 'curr_edge'
        if self.edge_queried == 1:
            if action_name == 'curr_edge':
                action_name = 'past_edge'
        if self.cloud_queried == 1:
            if action_name == 'curr_cloud':
                action_name = 'past_cloud'
        return action_name

    def _simulate_action_updates(self, action_name):
        # decision stage is defined as edge_queried,cloud_queried
        # decision stage 00, just see new sensory input, all actions allowed
        # decision stage 10, curr_edge disallowed
        # decision stage 01, curr_cloud disallowed
        # decision stage 11, curr_edge, curr_cloud disallowed
        # can also add the L2 difference between successive inputs x as a feature to RL
        stage_end = True
        # get the prediction and confidence depending on the action used
        # first two are stale measurements
        if action_name == 'past_edge':
            # WAS ERROR HERE, SHOULD BE -1 for both
            past_edge_predict = self.state_dict['past_edge_predict'][0]
            past_edge_conf = self.state_dict['past_edge_conf'][0]
            
            # update state
            self.state_dict['past_overall_predict'] = [past_edge_predict, past_edge_conf]
        elif action_name == 'past_cloud':
            # WAS ERROR HERE, SHOULD BE -1 for both
            past_cloud_predict = self.state_dict['past_cloud_predict'][0]
            past_cloud_conf = self.CLOUD_CONF
            
            # update state
            self.state_dict['past_overall_predict'] = [past_cloud_predict, past_cloud_conf]
        elif action_name == 'curr_edge':
            self.edge_queried = 1
            self.state_dict['edge_queried'] = [self.edge_queried]

            # query edge model
            curr_edge_prediction_vec = [self.edge_prediction_vec[self.t], self.edge_confidence_vec[self.t]]
            
            # update state
            self.state_dict['past_overall_predict'] = curr_edge_prediction_vec
            self.state_dict['past_edge_predict'] = [curr_edge_prediction_vec[0]]
            #self.state_dict = shift_dict_value(input_dict=self.state_dict,
            #                                   key='past_edge_predict_vector',
            #                                   shift_amount=-1,
            #                                   fill_value=curr_edge_prediction_vec[0])
            self.state_dict['past_edge_conf'] = [curr_edge_prediction_vec[1]]
            #self.state_dict = shift_dict_value(input_dict=self.state_dict,
            #                                   key='past_edge_conf_vector',
            #                                   shift_amount=-1,
            #                                   fill_value=curr_edge_prediction_vec[1])
            self.state_dict['past_edge_x'] = [self.state_dict['curr_query_x'][0]]
            #self.state_dict = shift_dict_value(input_dict=self.state_dict,
            #                                   key='past_edge_x_vector',
            #                                   shift_amount=-1,
            #                                   fill_value=self.state_dict['curr_query_x'][0])
            self.state_dict['past_edge_xdiff'] = [0.]
            self.state_dict['past_edge_tdiff'] = [0.]
            self.state_dict['past_edge_query_time'] = [self.t]
            stage_end = False
        elif action_name == 'curr_cloud':
            self.cloud_queried = 1
            self.state_dict['cloud_queried'] = [self.cloud_queried]
            # update num remaining queries
            self.num_queries_remain -= 1
            self.state_dict['num_queries_remain'] = [self.num_queries_remain]

            # 1.0 is the default cloud confidence
            curr_cloud_prediction_vec = [self.cloud_prediction_vec[self.t], self.CLOUD_CONF]

            # update state
            self.state_dict['past_overall_predict'] = curr_cloud_prediction_vec
            self.state_dict['past_cloud_predict'] = [curr_cloud_prediction_vec[0]]
            #self.state_dict = shift_dict_value(input_dict=self.state_dict,
            #                                   key='past_cloud_predict_vector',
            #                                   shift_amount=-1,
            #                                   fill_value=curr_cloud_prediction_vec[0])
            self.state_dict['past_cloud_x'] = [self.state_dict['curr_query_x'][0]]
            #self.state_dict = shift_dict_value(input_dict=self.state_dict,
            #                                   key='past_cloud_x_vector',
            #                                   shift_amount=-1,
            #                                   fill_value=self.state_dict['curr_query_x'][0])
            self.state_dict['past_cloud_xdiff'] = [0.]
            self.state_dict['past_cloud_tdiff'] = [0.]
            self.state_dict['past_cloud_query_time'] = [self.t]
            stage_end = False
        return stage_end

    def _compute_reward(self, action, stage_end, y_true_input=None):
        # weights for accuracy, probes
        query_cost_weight = self.reward_params_dict['query_cost_weight']
        accuracy_weight = self.reward_params_dict['accuracy_weight']
        query_x = self.state_dict['curr_query_x'][0]
        prediction_confidence = self.state_dict['past_overall_predict']
        prediction = prediction_confidence[0]

        # add some noise to this
        query_noise = 0.0
        self.query_cost_term = self.query_cost_dict[action] + query_noise

        # only penalize the cost of probes
        reward = - query_cost_weight * self.query_cost_term
        self.accuracy_cost_term = 0.0
        y_true = np.nan
        # also penalize accuracy, full reward
        if stage_end:
            
            y_true = y_true_input 
            #percent_error = calc_error_metrics(y_predict=prediction, y_true=y_true)
            #self.accuracy_cost_term = percent_error
            if prediction == y_true:
                self.accuracy_cost_term = 0.0
            else:
                self.accuracy_cost_term = 1.0
            #print(' ')
            #print('accuracy: ', self.accuracy_cost_term)
            #print('y_true: ', y_true)
            #print('prediction: ', prediction)
            #print(' ')
            reward -= accuracy_weight * self.accuracy_cost_term

        if self.print_mode:
            print(' ')
            print('reward: ', reward)
            print('query_cost_term: ', query_cost_term)
            print('accuracy_cost_term: ', accuracy_cost_term)
            print(' ')
        return reward, y_true

