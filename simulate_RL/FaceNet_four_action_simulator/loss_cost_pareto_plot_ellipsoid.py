# draw the classification loss vs offloading cost as a pareto optimal plot, with ellipsoids for covariance

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
import sys, os
import numpy as np
import itertools
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--RL_present', type=str, required=False,
                        default='both')
    parser.add_argument('--prefix', type=str, required=False,
                        default='v1Sim')
    parser.add_argument('--base-results-dir', type=str, required=True, default=None)
    return parser.parse_args()

# CITATION:
# the below sub-function is from https://scipython.com/book/chapter-7-matplotlib/examples/bmi-data-with-confidence-ellipses/

def get_cov_ellipse(cov, centre, nstd, **kwargs):
    """
    Return a matplotlib Ellipse patch representing the covariance matrix
    cov centred at centre and scaled by the factor nstd.

    """

    # Find and sort eigenvalues and eigenvectors into descending order
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # The anti-clockwise angle to rotate our ellipse by 
    vx, vy = eigvecs[:,0][0], eigvecs[:,0][1]
    theta = np.arctan2(vy, vx)

    # Width and height of ellipse to draw
    width, height = 2 * nstd * np.sqrt(eigvals)
    return Ellipse(xy=centre, width=width, height=height,
                   angle=np.degrees(theta), **kwargs)


if __name__ == '__main__':

    DATA_DELETE_MODE = True

    args = parse_args()
    # 'RL', 'baseline', or 'both'    
    RL_present = args.RL_present
    prefix = args.prefix
    base_log_path = args.base_results_dir

    HUGE_NEGATIVE = -1000000

    if DATA_DELETE_MODE:
        base_plot_dir = base_log_path + '/ELLIPSE_' + prefix + '/'
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
    query_cost_mean_latex = r'Mean Offloading Cost'  
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

    # 'Mean Offloading Cost', 'Mean Classification Loss', 'Offload Policy'
    color_list = ['red', 'blue', 'green', 'purple', 'orange', 'grey'] 

    x_var = 'Offload Policy' 

    actions = ['curr_edge', 'curr_cloud'] 
    order_list = ['random'] + actions + threshold_controller_list + ['RL', 'pure_oracle']
    latex_order_list = [remap_name_dict[x] for x in order_list]

    # THIS IS THE KEY COVARIANCE PLOT
    ########################################
    fig, ax = plt.subplots()
    handle_list = []
    alpha = 0.5

    for color_val, controller_name in zip(color_list, latex_order_list):
        controller_df = controller_rewards_df[controller_rewards_df[x_var] == controller_name]

        query_cost_col = list(controller_df[query_cost_mean_latex])

        accuracy_cost_column = list(controller_df[accuracy_cost_mean_latex])

        query_cost_mean = np.mean(query_cost_col)
        accuracy_cost_mean = np.mean(accuracy_cost_column)

        print(' ')
        print('controller_name: ', controller_name)
        print(query_cost_mean, accuracy_cost_mean)
        print(' ')
        
        query_cost_column = query_cost_col

        print(np.std(query_cost_column))
        cov = np.cov(query_cost_column, accuracy_cost_column)

        #r = sns.color_palette(color_val)[2]
        r = color_val

        patch = mpatches.Patch(color=r, label=controller_name)
        handle_list.append(patch)

        scaled_cov = cov/len(query_cost_col)    

        e = get_cov_ellipse(scaled_cov, (query_cost_mean, accuracy_cost_mean), 1.96,
                            fc=r, alpha=alpha)

        ax.add_artist(e)

        # the mean dot
        ax.plot([query_cost_mean], [accuracy_cost_mean], 'o', color=r, markersize = 6.0)


    Y_SCALE = 0.4
    ax.set_xlim(0, 3)
    ax.set_ylim(0, Y_SCALE*1)
    ax.set_xlabel(query_cost_mean_latex)
    ax.set_ylabel(accuracy_cost_mean_latex)
    ax.legend(loc='upper left')

    ax.text(0.5, 0.25, 'All-Robot', fontsize=15)
    ax.text(1.6, 0.3, 'Random', fontsize=15)
    ax.text(1.6, 0.15, 'Heuristic', fontsize=15)
    ax.text(2.5, 0.05, 'All-Cloud', fontsize=15)
    ax.text(0.6, 0.05, 'RL', fontsize=15)
    #ax.text(0.75, 0.1, 'Oracle', fontsize=15)

    plt.legend(handles=handle_list)
    plt.savefig(base_plot_dir + '/scaled_ellipse_noshade.png')
    plt.close()

    ## SCATTERPLOT  [NOT PRESENT IN PAPER]
    #########################################
    #fig, ax = plt.subplots()
    #handle_list = []
    #alpha = 0.5

    #for color_val, controller_name in zip(color_list, latex_order_list):
    #    controller_df = controller_rewards_df[controller_rewards_df[x_var] == controller_name]

    #    query_cost_col = list(controller_df[query_cost_mean_latex])

    #    accuracy_cost_column = list(controller_df[accuracy_cost_mean_latex])

    #    query_cost_mean = np.mean(query_cost_col)
    #    accuracy_cost_mean = np.mean(accuracy_cost_column)

    #    # scatterplot

    #    print(' ')
    #    print('controller_name: ', controller_name)
    #    print(query_cost_mean, accuracy_cost_mean)
    #    print(' ')
    #    
    #    query_cost_column = query_cost_col

    #    #r = sns.color_palette(color_val)[2]
    #    r = color_val

    #    patch = mpatches.Patch(color=r, label=controller_name)
    #    handle_list.append(patch)

    #    ax.scatter(query_cost_column, accuracy_cost_column, alpha, color=r)
    #    # the mean dot
    #    ax.plot([query_cost_mean], [accuracy_cost_mean], 'o', color=r, markersize = 6.0)

    #ax.set_xlim(0, 3)
    #ax.set_ylim(0, 1)
    #ax.set_xlabel(query_cost_mean_latex)
    #ax.set_ylabel(accuracy_cost_mean_latex)
    #ax.legend(loc='upper left')

    #ax.text(1.0, 0.65, 'Heuristic', fontsize=15)
    #ax.text(1.35, 0.9, 'Random', fontsize=15)
    #ax.text(0.3, 0.8, 'All-Robot', fontsize=15)
    #ax.text(2.3, 0.2, 'All-Cloud', fontsize=15)
    #ax.text(0.3, 0.25, 'RL', fontsize=15)
    #ax.text(0.75, 0.1, 'Oracle', fontsize=15)

    #plt.legend(handles=handle_list)
    #plt.savefig(base_plot_dir + '/scatter_ellipse_noshade.png')
    #plt.close()
