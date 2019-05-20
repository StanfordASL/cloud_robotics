import numpy as np
import sys,os

CLOUD_ROOT_DIR=os.environ['CLOUD_ROOT_DIR']
sys.path.append(CLOUD_ROOT_DIR)
sys.path.append(CLOUD_ROOT_DIR + '/utils/')

from textfile_utils import *
from plotting_utils import *
from collections import OrderedDict

def get_bw_info(bw_file_names = None, BASE_ROOT_DIR = None):

    total_bw_vector = []
    legend_vec = []

    for k, v in bw_file_names.items():
            print(' ')
            print('k: ', k)
            print('v: ', v)

            fname = BASE_ROOT_DIR + '/' + k

            bw_vector = []
            with open(fname, 'r') as f:
                for line in f:
                    try:
                        if line.startswith('average'):
                            bw = line.split()[-1]
                            bw_val = float(bw.strip()[:-4])
                            # to make in Mbps
                            bw_vector.append(8*bw_val)
                    except:
                        pass
            print('mean: ', np.median(bw_vector))
            print('std: ', np.std(bw_vector))
            print(' ')

            total_bw_vector.append(bw_vector)
            legend_vec.append(v)

    return total_bw_vector, legend_vec


def get_hz_info(hz_file_names = None, BASE_ROOT_DIR = None):

    ##################################
    print('HERTZ')
    total_hz_vector = []
    legend_vec = []

    # parse the hertz data 
    for k, v in hz_file_names.items():
            print(' ')
            print('k: ', k)
            print('v: ', v)

            fname = BASE_ROOT_DIR + '/' + k

            hz_vector = []
            with open(fname, 'r') as f:
                for line in f:
                    try:
                        if line.startswith('average'):
                            hz_val = float(line.split()[-1])
                            hz_vector.append(hz_val)
                    except:
                        pass
            print('mean: ', np.mean(hz_vector))
            print('std: ', np.std(hz_vector))
            print(' ')

            total_hz_vector.append(hz_vector)
            legend_vec.append(v)

    return total_hz_vector, legend_vec

if __name__ == '__main__':
        
        BASE_ROOT_DIR = 'camera_ready_experiments/'
        BASE_PLOT_DIR = 'camera_ready_plots/'

        prefix_list = ['with_receiver', 'with_rviz', 'with_download', 'lidar1', 'lidar2']

        for prefix in prefix_list:

            bw_file_names = OrderedDict()
            bw_file_names['bw_lidar_source_with_receiver.txt']= 'ROS LIDAR source'
            bw_file_names['bw_lidar_receiver_' + str(prefix) + '.txt'] = 'ROS LIDAR receiver'

            hz_file_names = OrderedDict()
            hz_file_names['hz_lidar_source_with_receiver.txt'] = 'ROS LIDAR source'
            hz_file_names['hz_lidar_receiver_' + str(prefix) + '.txt'] = 'ROS LIDAR receiver'

            LIDAR_plot_file = BASE_PLOT_DIR + '/' + prefix + '_LIDAR_overlaid_final.png'

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
            ax1.set_ylim([0,.5])
            ax2.set_ylim([0,3])
            # Generate a random univariate dataset
            sns.distplot(total_bw_vector[0], norm_hist = norm, ax = ax1, color = 'blue')
            sns.distplot(total_bw_vector[1], norm_hist = norm, ax = ax1, color = 'red')
            #ax1.hist(total_bw_vector[0], density = True)
            #ax1.hist(total_bw_vector[1], density = True)

            
            plt.hold(True)
            xlabel = 'Bandwidth (Mbps)'        
            ax1.set_xlabel(xlabel)
            title_str = 'LIDAR Bandwidth: ' + prefix
            ax1.set_title(title_str)

            ###################################
            # Generate a random univariate dataset

            sns.distplot(total_hz_vector[0], norm_hist = norm, ax = ax2, color = 'blue')
            sns.distplot(total_hz_vector[1], norm_hist = norm, ax = ax2, color = 'red')
            #ax2.hist(total_hz_vector[0], density = True)
            #ax2.hist(total_hz_vector[1], density = True)
     
            plt.hold(True)
            xlabel = 'Sampling Rate (Hz)'        
            ax2.set_xlabel(xlabel)
            #title_str = 'LIDAR Sampling Rate Distribution'
            #ax2.set_title(title_str)
            ax1.legend(['LIDAR receiver', 'LIDAR source'], loc='best')

            #plt.setp(axes)
            plt.tight_layout()

            plt.savefig(LIDAR_plot_file)
            plt.close()
            #Creates two subplots and unpacks the output array immediately
