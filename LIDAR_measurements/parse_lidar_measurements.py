import numpy as np
import sys,os

CLOUD_ROOT_DIR=os.environ['CLOUD_ROOT_DIR']
sys.path.append(CLOUD_ROOT_DIR)
sys.path.append(CLOUD_ROOT_DIR + '/utils/')

from textfile_utils import *
from plotting_utils import *

if __name__ == '__main__':
        
        BASE_ROOT_DIR = 'submission_experiments/'
        BASE_PLOT_DIR = 'submission_plots/'

        bw_file_names = {'received_lidar_bw_wo_echo.txt': 'ROS LIDAR source',  
                        'received_lidar_bw_with_echo.txt': 'ROS LIDAR receiver'}

        hz_file_names = {'received_lidar_hz_wo_echo.txt': 'ROS LIDAR source', 
                        'received_lidar_hz_with_echo.txt': 'ROS LIDAR receiver'}

        # parse the bandwidth data 
        total_bw_vector = []
        legend_vec = []

        bw_plot_fname = BASE_PLOT_DIR + '/bw_LIDAR.pdf'

        for k, v in bw_file_names.items():
                print(' ')
                print('k: ', k)
                print('v: ', v)

                fname = BASE_ROOT_DIR + '/' + k

                bw_vector = []
                with open(fname, 'r') as f:
                    for line in f:
                        if line.startswith('average'):
                            bw = line.split()[-1]
                            bw_val = float(bw.strip()[:-4])
                            # to make in Mbps
                            bw_vector.append(8*bw_val)
                
                print('mean: ', np.median(bw_vector))
                print('std: ', np.std(bw_vector))
                print(' ')

                total_bw_vector.append(bw_vector)
                legend_vec.append(v)

        title_str = 'LIDAR Bandwidth Distribution'
        #plot_several_pdf(data_vector_list = total_bw_vector, xlabel = 'Bandwidth (MB/s)', plot_file = bw_plot_fname, title_str = title_str, legend = legend_vec)


        ##################################
        print('HERTZ')
        total_hz_vector = []
        legend_vec = []

        hz_plot_fname = BASE_PLOT_DIR + '/hz_LIDAR.pdf'
        # parse the hertz data 
        for k, v in hz_file_names.items():
                print(' ')
                print('k: ', k)
                print('v: ', v)

                fname = BASE_ROOT_DIR + '/' + k

                hz_vector = []
                with open(fname, 'r') as f:
                    for line in f:
                        if line.startswith('average'):
                            hz_val = float(line.split()[-1])
                            hz_vector.append(hz_val)
                
                print('mean: ', np.mean(hz_vector))
                print('std: ', np.std(hz_vector))
                print(' ')

                total_hz_vector.append(hz_vector)
                legend_vec.append(v)


        #plot_several_pdf(data_vector_list = total_hz_vector, xlabel = 'Sampling Rate (Hz)', plot_file = hz_plot_fname, title_str = title_str, legend = legend_vec)

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
        sns.distplot(total_bw_vector[0], norm_hist = norm, ax = ax1)
        sns.distplot(total_bw_vector[1], norm_hist = norm, ax = ax1)
        #ax1.hist(total_bw_vector[0], density = True)
        #ax1.hist(total_bw_vector[1], density = True)

        
        plt.hold(True)
        xlabel = 'Bandwidth (Mbps)'        
        ax1.set_xlabel(xlabel)
        title_str = 'LIDAR Bandwidth Distribution'
        #ax1.set_title(title_str)

        ###################################
        # Generate a random univariate dataset
        sns.distplot(total_hz_vector[0], norm_hist = norm, ax = ax2)
        sns.distplot(total_hz_vector[1], norm_hist = norm, ax = ax2)
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

        LIDAR_plot_file = BASE_PLOT_DIR + '/LIDAR_overlaid_final.png'
        plt.savefig(LIDAR_plot_file)
        #Creates two subplots and unpacks the output array immediately
