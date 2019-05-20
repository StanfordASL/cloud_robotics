import numpy as np
import sys,os

CLOUD_ROOT_DIR=os.environ['CLOUD_ROOT_DIR']
sys.path.append(CLOUD_ROOT_DIR)
sys.path.append(CLOUD_ROOT_DIR + '/utils/')

from textfile_utils import *
from plotting_utils import *
from collections import OrderedDict
from utils_parser import *

if __name__ == '__main__':
        
        BASE_ROOT_DIR = 'camera_ready_experiments/'
        BASE_PLOT_DIR = 'several_sender_camera_ready_plots/'

        prefix_list = ['with_receiver', 'with_download', 'both_lidars']

        #final_legend_vec = ['LIDAR Source', '1 Receiver', '1 Receiver with Heavy Background Traffic', '2 Concurrent LIDAR Pairs']
        final_legend_vec = ['At LIDAR Source', 'At 1 LIDAR Recv.', '1 Recv. + Heavy Traffic', '2 LIDAR Recv.']

        bw_file_names = OrderedDict()
        bw_file_names['bw_lidar_source_with_receiver.txt']= 'ROS LIDAR source'

        hz_file_names = OrderedDict()
        hz_file_names['hz_lidar_source_with_receiver.txt'] = 'ROS LIDAR source'

        for prefix in prefix_list:

            bw_file_names['bw_lidar_receiver_' + str(prefix) + '.txt'] = 'ROS LIDAR receiver'

            hz_file_names['hz_lidar_receiver_' + str(prefix) + '.txt'] = 'ROS LIDAR receiver'


        LIDAR_plot_file = BASE_PLOT_DIR + '/SEVERAL_EXP_LIDAR_overlaid_final.png'

        # total bw_vector
        ##################################
        total_bw_vector, legend_vec = get_bw_info(bw_file_names = bw_file_names, BASE_ROOT_DIR = BASE_ROOT_DIR)

        total_hz_vector, legend_vec = get_hz_info(hz_file_names = hz_file_names, BASE_ROOT_DIR = BASE_ROOT_DIR)

        norm = True
        f, axes = plt.subplots(1, 2, sharey=True)
        f, axes = plt.subplots(1, 2)
        #sns.despine(left=True)

        ax1 = axes[0]
        ax2 = axes[1]
        
        ylabel = 'Density'
        ax1.set_ylabel(ylabel)

        ylabel = 'Density'
        ax1.set_ylabel(ylabel)
        ax2.set_ylabel(ylabel)
        # Generate a random univariate dataset
        NORMED = True
        for i in range(len(total_bw_vector)):
            #sns.distplot(total_bw_vector[i], norm_hist = norm, ax = ax1)
            ax1.hist(total_bw_vector[i], normed = NORMED)
        #ax1.hist(total_bw_vector[1], density = True)

        
        plt.hold(True)
        xlabel = 'Bandwidth (Mbps)'        
        ax1.set_xlabel(xlabel)
        #title_str = 'LIDAR Bandwidth: ' + prefix
        #ax1.set_title(title_str)

        ###################################
        # Generate a random univariate dataset
        for i in range(len(total_hz_vector)):
            #sns.distplot(total_hz_vector[i], norm_hist = norm, ax = ax1)
            ax2.hist(total_hz_vector[i], normed = NORMED)

        #sns.distplot(total_hz_vector[0], norm_hist = norm, ax = ax2, color = 'blue')
        #sns.distplot(total_hz_vector[1], norm_hist = norm, ax = ax2, color = 'red')
        #ax2.hist(total_hz_vector[0], density = True)
        #ax2.hist(total_hz_vector[1], density = True)
 
        plt.hold(True)
        xlabel = 'Sampling Rate (Hz)'        
        ax2.set_xlabel(xlabel)
        #title_str = 'LIDAR Sampling Rate Distribution'
        #ax2.set_title(title_str)
        ax2.legend(final_legend_vec, loc='best')

        #plt.setp(axes)
        plt.tight_layout()

        ax1.set_ylim([0,0.3])
        ax1.set_xlim([0,100])
        ax2.set_ylim([0,3])
        ax2.set_xlim([0,15])
        plt.savefig(LIDAR_plot_file)
        plt.close()
        #Creates two subplots and unpacks the output array immediately
