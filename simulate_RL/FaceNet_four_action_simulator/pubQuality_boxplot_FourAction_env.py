# draw a boxplot of RL vs benchmark controller performance

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

try: 
    import ConfigParser
except:
    import configparser

CLOUD_ROOT_DIR=os.environ['CLOUD_ROOT_DIR']
sys.path.append(CLOUD_ROOT_DIR)

UTILS_DIR = CLOUD_ROOT_DIR + '/utils/'
sys.path.append(UTILS_DIR)

from textfile_utils import *
from plotting_utils import *
from RSS_data_parsing_utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--RL_present', type=str, required=False,
                        default='both')
    parser.add_argument('--prefix', type=str, required=False,
                        default='v1Sim')
    parser.add_argument('--base-results-dir', type=str, required=True, default=None)
    return parser.parse_args()

if __name__ == '__main__':

    DATA_DELETE_MODE = True

    args = parse_args()
    # 'RL', 'baseline', or 'both'    
    RL_present = args.RL_present
    prefix = args.prefix
    base_log_path = args.base_results_dir

    HUGE_NEGATIVE = -1000000

    if DATA_DELETE_MODE:
        base_plot_dir = base_log_path + '/boxplot_' + prefix + '/'
        remove_and_create_dir(base_plot_dir)

    # load the results from the RL runs and the baseline runs 
    RL_episode_csv_path = base_log_path + '/RL_results_df.csv'
    baseline_controller_episode_csv_path = base_log_path + '/' + 'FourAction_FaceNet_baseline_data_' + prefix + '/FourAction_episode_results.csv'

    # cat them into one big dataframe
    controller_rewards_df, _ = join_RL_baseline_controllers(RL_episode_csv_path = RL_episode_csv_path, baseline_controller_episode_csv_path = baseline_controller_episode_csv_path, RL_present = RL_present)

    # write a summary of the median reward per controller
    # this is used for reporting in the final paper
    print('SUMMARY REWARDS STATS')
    print(controller_rewards_df.groupby('controller_name')['reward_sum'].median())

    median_reward_df = controller_rewards_df.groupby('controller_name')['reward_mean'].median()
    median_cost_df = controller_rewards_df.groupby('controller_name')['query_cost_mean'].median()
    median_loss_df = controller_rewards_df.groupby('controller_name')['accuracy_cost_mean'].median()

    # write median rewards, cost, accuracy loss to a file
    ############################
    out_file = base_plot_dir + '/fold.results.txt'

    metrics_list = ['reward', 'cost', 'loss']
    metric_dict_list = [dict(median_reward_df), dict(median_cost_df), dict(median_loss_df)]

    with open(out_file, 'w') as f:
        for metric, local_dict in zip(metrics_list, metric_dict_list):
            f.write('\n')
            RL_value = local_dict['RL']
            for controller, value in local_dict.items(): 
                if (value < 0)  and (RL_value < 0):
                        fold = value/RL_value
                else:
                        fold = RL_value/value

                out_str = '\t'.join([metric, controller, str(fold)]) 
                f.write(out_str + '\n')
   
    ############################
 
    # we have simulated several threshold based controllers, loop over their rewards to get the best one
    ############################
    median_rewards_df = controller_rewards_df.groupby('controller_name')['reward_sum'].median()
    best_threshold_controller = None
    best_threshold_reward = HUGE_NEGATIVE
    for name, reward in median_rewards_df.iteritems():
        if name.startswith('threshold'):
            if reward > best_threshold_reward:
                best_threshold_controller = name
                best_threshold_reward = reward

    print('best_threshold_controller', best_threshold_controller)
    print('best_threshold_reward', best_threshold_reward)

    # controller_rewards_df
    ############################
    # ['Unnamed: 0', 'accuracy_cost_mean', 'accuracy_cost_sum', 'controller_name', 'episode', 'query_cost_mean', 'query_cost_sum', 'reward_mean', 'reward_sum']

    reward_mean_latex = r'Mean Episode Reward'
    query_cost_mean_latex = r'Mean Model Query Cost'  
    accuracy_cost_mean_latex = r'Mean Classification Loss'

    # extract the controller names and map them to the policy name to display in LaTeX
    ############################
    controller_names_list = list(set(controller_rewards_df['controller_name']))
    controller_rewards_df[reward_mean_latex] = controller_rewards_df['reward_mean']
    controller_rewards_df[query_cost_mean_latex] = controller_rewards_df['query_cost_mean']
    controller_rewards_df[accuracy_cost_mean_latex] = controller_rewards_df['accuracy_cost_mean']

    # map the simple names in the dataframe to LaTex display names for the policy plots
    remap_name_dict = {}
    remap_name_dict['random'] = r'$\pi_{\mathtt{offload}}^{\mathtt{random}}$'
    remap_name_dict['past_edge'] = r'$\pi_{\mathtt{offload}}^{\mathtt{past-robot}}$'
    remap_name_dict['curr_edge'] = r'$\pi_{\mathtt{offload}}^{\mathtt{all-robot}}$'
    remap_name_dict['past_cloud'] = r'$\pi_{\mathtt{offload}}^{\mathtt{past-cloud}}$'
    remap_name_dict['curr_cloud'] = r'$\pi_{\mathtt{offload}}^{\mathtt{all-cloud}}$'
    remap_name_dict['RL'] = r'$\pi_{\mathtt{offload}}^{\mathtt{RL}}$'
    remap_name_dict['oracle'] = r'$\pi_{\mathtt{offload}}^{\mathtt{semiOracle1}}$'
    remap_name_dict['pure_oracle'] = r'$\pi_{\mathtt{offload}}^{\mathtt{Oracle}}$'

    # get the policy name for the threshold heuristic controllers by extracting their threshold
    threshold_val_list = []
    for ctrller in controller_names_list:
        if ctrller.startswith('threshold'):
            threshold_val = ctrller.split('-')[1]
            #remap_name_dict[ctrller] = r'$\pi_{\mathtt{offload}}^{\mathtt{thresh-' + str(threshold_val) + '}}$'
            remap_name_dict[ctrller] = r'$\pi_{\mathtt{offload}}^{\mathtt{heuristic}}$'
            threshold_val_list.append(threshold_val)

    threshold_val_list.sort()
    threshold_controller_list = ['threshold-' + str(x) for x in threshold_val_list]
    threshold_controller_list = [best_threshold_controller]

    print(remap_name_dict)
    controller_rewards_df['Offload Policy'] = [remap_name_dict[x] for x in controller_rewards_df['controller_name']]

    # plot paired boxplot across all policies
    ############################
    y_var_list = ['reward_mean', 'query_cost_mean', 'accuracy_cost_mean']
    y_var_list = [reward_mean_latex, query_cost_mean_latex, accuracy_cost_mean_latex]

    # what fixed controllers to compare with
    actions = ['curr_edge', 'curr_cloud'] 

    # plot reward, cost, accuracy loss 
    for i, y_var in enumerate(y_var_list):

        ## baseline results
        ########################################################
        x_var = 'Offload Policy' 
        plot_fname = base_plot_dir + '/FourAction_baseline_' + str(i) + '.png' 
        title_str = ' '

        order_list = ['random'] + actions + threshold_controller_list + ['pure_oracle']
        latex_order_list = [remap_name_dict[x] for x in order_list]

        rmin = 1.1 * controller_rewards_df[y_var].quantile(0.05)
        rmax = 0.1 * controller_rewards_df[y_var].quantile(0.99)

        ylim = [rmin, rmax]
        ylim = None

	# colors for policies: RL is in orange, oracle in gray
        pal = ['red', 'blue', 'green', 'purple', 'orange', 'grey']
        plot_grouped_boxplot(df = controller_rewards_df, x_var = x_var, y_var = y_var, plot_file = plot_fname, ylim = ylim, title_str = title_str, order_list = latex_order_list, pal = pal)


        ## RL
        ########################################################
        if 'RL' in controller_names_list: 
            plot_fname = base_plot_dir + '/RL_' + str(i) + '.png' 
            order_list = ['random'] + actions + threshold_controller_list + ['RL', 'pure_oracle']
            latex_order_list = [remap_name_dict[x] for x in order_list]

            plot_grouped_boxplot(df = controller_rewards_df, x_var = x_var, y_var = y_var, plot_file = plot_fname, ylim = ylim, title_str = title_str, order_list = latex_order_list, pal = pal)
