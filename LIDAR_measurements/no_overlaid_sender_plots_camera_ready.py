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
        final_legend_vec = ['LIDAR Source', '1 Recv.', '1 Recv. + Heavy Traffic', '2 LIDARs']

        bw_file_names = OrderedDict()
        bw_file_names['bw_lidar_source_with_receiver.txt']= 'ROS LIDAR source'

        hz_file_names = OrderedDict()
        hz_file_names['hz_lidar_source_with_receiver.txt'] = 'ROS LIDAR source'

        for prefix in prefix_list:

            bw_file_names['bw_lidar_receiver_' + str(prefix) + '.txt'] = 'ROS LIDAR receiver'

            hz_file_names['hz_lidar_receiver_' + str(prefix) + '.txt'] = 'ROS LIDAR receiver'


        type_plot = 'bw'

        LIDAR_plot_file = BASE_PLOT_DIR + '/' + type_plot + 'SEVERAL_EXP_LIDAR_overlaid_final.png'

        # total bw_vector
        ##################################
        total_bw_vector, legend_vec = get_bw_info(bw_file_names = bw_file_names, BASE_ROOT_DIR = BASE_ROOT_DIR)

        total_hz_vector, legend_vec = get_hz_info(hz_file_names = hz_file_names, BASE_ROOT_DIR = BASE_ROOT_DIR)

        if type_plot == 'hz':
            vector_to_plot = total_hz_vector
            xlabel = 'Sampling Rate (Hz)'       
            xlim = [0, 14]
            ylim = [0, 3]
        else:
            vector_to_plot = total_bw_vector
            xlabel = 'Bandwidth (Mbps)'        
            xlim = [0, 100]
            ylim = [0, 0.3]

        norm = True
        f, ax1 = plt.subplots()

        
        ylabel = 'Density'
        ax1.set_ylabel(ylabel)

        # Generate a random univariate dataset
        NORMED = True
        for i in range(len(total_bw_vector)):
            #sns.distplot(total_bw_vector[i], norm_hist = norm, ax = ax1)
            ax1.hist(total_bw_vector[i], normed = NORMED)

        
        ax1.set_xlabel(xlabel)
        ax1.legend(final_legend_vec, loc='best')

        #plt.setp(axes)
        plt.tight_layout()

        ax1.set_ylim(ylim)
        ax1.set_xlim(xlim)
        plt.savefig(LIDAR_plot_file)
        plt.close()
