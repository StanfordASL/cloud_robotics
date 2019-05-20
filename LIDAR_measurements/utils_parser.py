import numpy as np
import sys,os


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
                            #print(bw)

                            bw_val = float(bw.strip()[:-4])
                            if 'KB/s' in bw:
                                final_bw_val = bw_val/1000.0
                                #print(bw_val, final_bw_val)
                            else:
                                final_bw_val = bw_val

                            # to make in Mbps
                            bw_vector.append(8*final_bw_val)
                    except:
                        pass
            print('mean: ', np.median(bw_vector))
            print('std: ', np.std(bw_vector))
            print('len: ', len(bw_vector))
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
            print('len: ', len(hz_vector))
            print(' ')

            total_hz_vector.append(hz_vector)
            legend_vec.append(v)

    return total_hz_vector, legend_vec

