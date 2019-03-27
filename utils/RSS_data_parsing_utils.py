import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import random
import itertools
import sys,os
import copy
import pandas

from plotting_utils import *
from textfile_utils import *

"""
    load results from the env per episode, create a summary and then join RL performance with baseline controllers
"""

def join_RL_baseline_controllers(RL_episode_csv_path = 'RL_data/RL_results_df.csv', baseline_controller_episode_csv_path = 'AQE_data/sample_results.csv', RL_present = 'both'):

    # summary results df
    #['accuracy_cost_mean', 'accuracy_cost_sum', 'controller_name', 'episode', 'query_cost_mean', 'query_cost_sum', 'reward_mean', 'reward_sum', 'cloud_query_budget_frac']
   
    # per episode df
    #['Unnamed: 0', 'accuracy_cost_term', 'accuracy_weight', 'cloud_query_budget_frac', 'control_action', 'controller_name', 'episode_number', 'query_cost_term', 'query_cost_weight', 'reward', 't', 'y_pred', 'y_true']

    if RL_present == 'both':
        # first load RL dfs
        RL_per_episode_results_df = pandas.read_csv(RL_episode_csv_path)

        RL_summary_df = results_df_to_controller_rewards_df(results_df = RL_per_episode_results_df)
        
        # now load rest of dfs
        baseline_controller_per_episode_results_df = pandas.read_csv(baseline_controller_episode_csv_path)

        # columns of per episode df
        # ,accuracy_cost_term,accuracy_weight,cloud_query_budget_frac,control_action,controller_name,episode_number,query_cost_term,query_cost_weight,reward,t,y_pred,y_true
        baseline_controller_summary_df = results_df_to_controller_rewards_df(results_df = baseline_controller_per_episode_results_df)

        joint_RL_summary_df = baseline_controller_summary_df.append(RL_summary_df)
        joint_RL_episode_df = baseline_controller_per_episode_results_df.append(RL_per_episode_results_df)

        return joint_RL_summary_df, joint_RL_episode_df
    
    elif RL_present == 'baseline':
        # now load rest of dfs
        baseline_controller_per_episode_results_df = pandas.read_csv(baseline_controller_episode_csv_path)

        # columns of per episode df
        # ,accuracy_cost_term,accuracy_weight,cloud_query_budget_frac,control_action,controller_name,episode_number,query_cost_term,query_cost_weight,reward,t,y_pred,y_true
        baseline_controller_summary_df = results_df_to_controller_rewards_df(results_df = baseline_controller_per_episode_results_df)

        return baseline_controller_summary_df, baseline_controller_per_episode_results_df

    # 'RL' only
    else:
        # first load RL dfs
        RL_per_episode_results_df = pandas.read_csv(RL_episode_csv_path)

        RL_summary_df = results_df_to_controller_rewards_df(results_df = RL_per_episode_results_df)

        return RL_summary_df, RL_per_episode_results_df


def results_df_to_controller_rewards_df(results_df = None, controller_results_csv_fname = None):
    # list of columns: ['accuracy_cost_term', 'accuracy_weight', 'control_action', 'controller_name', 'episode_number', 'query_cost_term', 'query_cost_weight', 'reward', 't', 'y_pred', 'y_true']
    # aggregate these into a results df to plot
    # controller_name, sum reward, mean reward, sum query cost term, mean query cost term, sum accuracy cost term, sum accuracy cost weight
    controller_rewards_df = pandas.DataFrame()

    controller_name_list = list(set(results_df['controller_name']))

    for controller_name in controller_name_list:
        controller_df = results_df[results_df['controller_name'] == controller_name]

        row_dict = {}
        row_dict['controller_name'] = [controller_name]
    
        episode_list = list(set(controller_df['episode_number']))

        for episode in episode_list:
            specific_controller_episode_df = controller_df[controller_df['episode_number'] == episode]


            accuracy_weight = list(set(specific_controller_episode_df['accuracy_weight']))[0]
            query_cost_weight = list(set(specific_controller_episode_df['query_cost_weight']))[0]
            cloud_query_budget_frac = list(set(specific_controller_episode_df['cloud_query_budget_frac']))[0]

            row_dict['episode'] = [episode]
            row_dict['reward_mean'] = [specific_controller_episode_df['reward'].mean()]
            row_dict['reward_sum'] = [specific_controller_episode_df['reward'].sum()]

            row_dict['query_cost_mean'] = [specific_controller_episode_df['query_cost_term'].mean()]
            row_dict['query_cost_sum'] = [specific_controller_episode_df['query_cost_term'].sum()]
            row_dict['query_cost_weight'] = [query_cost_weight]

            row_dict['accuracy_cost_mean'] = [specific_controller_episode_df['accuracy_cost_term'].mean()]
            row_dict['accuracy_cost_sum'] = [specific_controller_episode_df['accuracy_cost_term'].sum()]
            row_dict['accuracy_weight'] = [accuracy_weight]
            row_dict['cloud_query_budget_frac'] = [cloud_query_budget_frac]

            row_df = pandas.DataFrame(row_dict)
            controller_rewards_df = controller_rewards_df.append(row_df)

    #print(controller_rewards_df[['controller_name', 'reward_mean']])

    if controller_results_csv_fname:
        controller_rewards_df.to_csv(controller_results_csv_fname)
    return controller_rewards_df

"""
    plot timeseries of an episode, specific to offloading!
"""

def plot_ts_episode(all_episode_df = None, episode_number = None, base_dir = None, episode_str = None, ylim = [-10, 5]):
   
    # get df for a specific episode
    df = all_episode_df[all_episode_df.episode_number == episode_number]

    # get a timeseries of the key colums and overlay them
    reward_ts = list(df.reward)
    print(reward_ts)

    accuracy_cost_ts = list(df.accuracy_cost_term)
    print(accuracy_cost_ts)

    action_ts = list(df.numeric_control_action)
    print(action_ts)

    edge_confidence_ts = list(df.edge_confidence)
    
    edge_cloud_accuracy_gap_ts = list(df.edge_cloud_accuracy_gap)

    y_pred_ts = list(df.y_pred)

    y_true_ts = list(df.y_true)

    cloud_query_budget_frac = list(set(df.cloud_query_budget_frac))[0]
    print('cloud_query_budget_frac: ', cloud_query_budget_frac)

    controller_name = list(set(df.controller_name))[0]
    print('controller_name: ', controller_name)

    accuracy_weight = list(set(df.accuracy_weight))[0]
    print('accuracy_weight: ', accuracy_weight)

    query_cost_weight = list(set(df.query_cost_weight))[0]
    print('query_cost_weight: ', query_cost_weight)

    # now plot an overlaid timeseries of all the quantities
    # PLOT 1
    #######################################################
    episode_info_str = ' '.join([r' ,  budget $\frac{N_{\mathrm{budget}}}{T}$ : ', str(cloud_query_budget_frac), ',', str(episode_str)])

    title_str = r'$\alpha_{\mathtt{accuracy}}=' + str(query_cost_weight) + r', \alpha_{\mathtt{cost}}=' + str(accuracy_weight) + '$' + episode_info_str

    LW = 3.0
    LS = '-'

    normalized_ts_dict = {}
    # key = ts_name, value is a dict, value = {'xvec': , 'ts_vector', 'lw', 'linestyle', 'color'}
    action_name = r'Action $a_{\mathtt{offload}}^t$'
    reward_name = r'Reward $R_{\mathtt{offload}}^t$'
    accuracy_cost_name = r'Loss $L(y^t, \hat{y}^t)$'

    normalized_ts_dict[action_name] = {'ts_vector': action_ts, 'lw': LW, 'linestyle': LS}
    normalized_ts_dict[reward_name] = {'ts_vector': reward_ts, 'lw': LW, 'linestyle': LS}
    normalized_ts_dict[accuracy_cost_name] = {'ts_vector': accuracy_cost_ts, 'lw': LW, 'linestyle': LS}
    
    plot_file = base_dir + '/RL_' + str(episode_number) + '_' +  str(episode_str) + '.pdf'
    overlaid_ts(normalized_ts_dict = normalized_ts_dict, title_str = title_str, plot_file = plot_file, ylabel = None, xlabel = r'time $t$', fontsize = 12, ylim = ylim)

    # PLOT 2
    #######################################################
    # edge confidence, edge cloud accuracy gap, action
    action_name = r'Action $a_{\mathtt{offload}}^t$'
    edge_confidence_name = r'Edge confidence ${\mathtt{conf}_\mathtt{edge}}^t$'
    edge_cloud_accuracy_gap_name = r'Edge cloud accuracy gap'

    normalized_ts_dict = {}
    # key = ts_name, value is a dict, value = {'xvec': , 'ts_vector', 'lw', 'linestyle', 'color'}
    normalized_ts_dict[action_name] = {'ts_vector': action_ts, 'lw': LW, 'linestyle': LS}
    normalized_ts_dict[edge_confidence_name] = {'ts_vector': edge_confidence_ts, 'lw': LW, 'linestyle': LS}
    normalized_ts_dict[edge_cloud_accuracy_gap_name] = {'ts_vector': edge_cloud_accuracy_gap_ts, 'lw': LW, 'linestyle': LS}
    
    plot_file = base_dir + '/conf_plot_RL_' + str(episode_number) + '_' +  str(episode_str) + '.pdf'
    overlaid_ts(normalized_ts_dict = normalized_ts_dict, title_str = title_str, plot_file = plot_file, ylabel = None, xlabel = r'time $t$', fontsize = 12, ylim = ylim)


