import pandas
import numpy
import sys,os
import numpy as np
from collections import OrderedDict
import argparse


base_video_dir = os.environ['CLOUD_ROOT_DIR']
utils_dir = base_video_dir + '/utils/'
sys.path.append(utils_dir)

from textfile_utils import *
from plotting_utils import * 

DEFAULT_OUTPUT_PKL_DIR= base_video_dir + '/data/video_data/'

if __name__ == "__main__":

    # arguments
    parser = argparse.ArgumentParser(description='DNNOffloader')
    parser.add_argument('--output-pkl-dir', type=str, required=False, default = DEFAULT_OUTPUT_PKL_DIR)

    args = parser.parse_args()
    OUTPUT_PKL_DIR = args.output_pkl_dir

    # START CODE
    ###############################
    all_video_annotations_dict = {}

    # VIDEO 1
    ###############################
    v1 = OrderedDict()

    v1['0-480'] = ['apoorva']
    v1['700-1170'] = ['james', 'apoorva']
    v1['1300-2120'] = ['joe', 'apoorva']

    all_video_annotations_dict['df_SVM_output_trial_1.txt'] = v1

    # VIDEO 2
    ###############################
    v2 = OrderedDict()

    v2['0-350'] = ['apoorva', 'sandeep']
    v2['600-1180'] = ['james']
    v2['1200-1680'] = ['sandeep', 'james']
    v2['1750-1900'] = ['sandeep']

    all_video_annotations_dict['df_SVM_output_trial_2.txt'] = v2

    # VIDEO 3
    ###############################
    v3 = OrderedDict()

    v3['0-220'] = ['apoorva']
    v3['330-495'] = ['james']
    v3['660-895'] = ['sandeep']

    all_video_annotations_dict['df_SVM_output_trial_3.txt'] = v3

    # VIDEO 4
    ###############################
    v4 = OrderedDict()

    v4['0-360'] = ['james']
    v4['450-840'] = ['apoorva']
    v4['960-20000'] = ['sandeep']

    all_video_annotations_dict['df_SVM_output_trial_4.txt'] = v4

    # VIDEO 5
    ###############################
    v5 = OrderedDict()
    v5['170-470'] = ['james', 'apoorva']
    v5['470-675'] = ['james']
    v5['675-890'] = ['apoorva']
    v5['890-1160'] = ['james', 'apoorva']

    all_video_annotations_dict['df_SVM_output_trial_5.txt'] = v5

    # VIDEO 6
    ###############################
    v6 = OrderedDict()
    v6['0-345'] = ['apoorva']
    v6['500-770'] = ['james']
    v6['770-1035'] = ['james', 'apoorva']
    v6['1190-1365'] = ['sandeep']

    all_video_annotations_dict['df_SVM_output_trial_6.txt'] = v6

    # VIDEO 7, sandeep abi
    ###############################
    v7 = OrderedDict()
    v7['0-220'] = ['abi']
    v7['220-350'] = ['sandeep', 'abi']
    v7['1000-1175'] = ['abi']
    v7['1500-20000'] = ['sandeep']

    all_video_annotations_dict['df_SVM_output_abi_sandeep.txt'] = v7


    # VIDEO 8, sandeep james apoorva
    ###############################


    v8 = OrderedDict()
    v8['0-195'] = ['apoorva']
    v8['195-250'] = ['apoorva', 'james']
    v8['250-600'] = ['apoorva', 'james']
    v8['600-1035'] = ['apoorva']
    v8['1035-2600'] = ['james']
    v8['2600-20000'] = ['sandeep']

    all_video_annotations_dict['df_SVM_output_james_apoorva_sandeep_1224.txt'] = v8

    ###############################
    video_annotations_pkl = OUTPUT_PKL_DIR + '/ground_truth_video_annotations.pkl'

    write_pkl(fname = video_annotations_pkl, input_dict = all_video_annotations_dict)

    data_dict = load_pkl(video_annotations_pkl)


