import bz2
import sys,os

import numpy as np
import os.path
#import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas

CLOUD_ROOT_DIR=os.environ['CLOUD_ROOT_DIR']
sys.path.append(CLOUD_ROOT_DIR)
sys.path.append(CLOUD_ROOT_DIR + '/utils/')

#from facenet_utils_csandeep import *
#from model import create_model

#from SVM_utils import *
from textfile_utils import *
import random
from plotting_utils import *
import copy
from collections import OrderedDict 
from calculation_utils import *


def sample_specific_face(train_test_df = None, seen_boolean = None, sampled_face_label = None, seed = None):
   
    all_potential_faces_df = train_test_df[train_test_df.true_label_name == sampled_face_label]

    #np.random.seed(seed)

    num_rows = all_potential_faces_df.shape[0]

    random_row_idx = np.random.choice(range(num_rows))

    random_row_df = all_potential_faces_df.iloc[random_row_idx]

    image_name = random_row_df['image_id']

    embedding_vector = np.array([float(x) for x in random_row_df['embedding_vector'].split('_')])

    edge_cloud_accuracy_gap = 1.0 - random_row_df['model_correct']

    edge_prediction = random_row_df['SVM_prediction_numeric']
    
    edge_prediction_name = random_row_df['SVM_prediction']

    edge_confidence = random_row_df['SVM_confidence']

    cloud_prediction = random_row_df['true_label_numeric']

    cloud_prediction_name = random_row_df['true_label_name']

    train_test_membership = random_row_df['train_test_membership']

    return image_name, embedding_vector, edge_cloud_accuracy_gap, edge_prediction, edge_confidence, cloud_prediction, train_test_membership, edge_prediction_name, cloud_prediction_name

"""
    stochastic ts based on facenet
"""

    # how to create a stochastic timeseries
    # choose a coherence time
    # choose from EITHER train or test based on flag
    # based on P_SEEN, choose from either SEEN or UNKNOWN
    # for each coherence time, choose a random label from SEEN, UNKNOWN, labels
    # choose random images for that label for the coherence time
    # populate edge prediction, edge confidence, cloud and the gap timeseries for all of them
    # repeat until done
    
    # plot, SEEN, UNSEEN, confidence etc timeseries as before
    # run the all-edge, all-cloud, and rest of benchmarks for the AQE simulator
    # see how it does on facenet

def facenet_stochastic_video(SVM_results_df = None, T = 200, coherence_time = 10, P_SEEN = 0.7, print_mode = False, seed = None, train_test_membership = 'TRAIN', EMBEDDING_DIM = 128, mini_coherence_time = 3, POISSON_MODE = True):

    edge_confidence_vec = []
    edge_prediction_vec = []
    cloud_prediction_vec = []
    true_value_vec = []
    edge_cloud_accuracy_gap_vec = []
    input_vec = []
    seen_vec = []
    rolling_diff_vec = []
    image_name_vec = []
    train_test_membership_vec = []
    embedding_norm_vec = []

    np.random.seed(seed)

    train_test_df = SVM_results_df[SVM_results_df['train_test_membership'] == train_test_membership]
    
    if print_mode:
        print('SVM: ', SVM_results_df.shape)
        print('train_test: ', train_test_df.shape)

    # all faces seen in this df
    seen_face_names = list(set(train_test_df[train_test_df['seen_unseen'] == 'SEEN']['true_label_name']))
    unseen_face_names = list(set(train_test_df[train_test_df['seen_unseen'] == 'UNSEEN']['true_label_name']))

    if print_mode:
        print('seen_face_names: ', seen_face_names)
        print('unseen_face_names: ', unseen_face_names)

    zero_embedding_vector = np.zeros(EMBEDDING_DIM)
    past_embedding_vector = zero_embedding_vector

    if POISSON_MODE:
        empirical_coherence_time = np.random.poisson(coherence_time-1) + 1
    else:
        empirical_coherence_time = coherence_time

    for t in range(T):
        # generate properties for the distro
        if t % empirical_coherence_time == 0:
           
            seen = get_random_uniform(p = P_SEEN)
            
            # depending on seen or not, get the face from the appropriate bin
            if seen:
                # a sample face
                sampled_face_label = np.random.choice(seen_face_names)
            else:
                # a sample face
                sampled_face_label = np.random.choice(unseen_face_names)

            if print_mode:
                print('t: ', t, 'sampled_face: ', sampled_face_label, 'seen: ', seen)

            if POISSON_MODE:
                empirical_coherence_time = np.random.poisson(coherence_time-1) + 1
            else:
                empirical_coherence_time = coherence_time
            
            #print('empirical_coherence_time', empirical_coherence_time)

        # for this face, see the feasible images, embeddings, and confidences we can choose from
        
        if t % mini_coherence_time == 0:
            image_name, embedding_vector, edge_cloud_accuracy_gap, edge_prediction, edge_confidence, cloud_prediction, train_test_membership, edge_prediction_name, cloud_prediction_name = sample_specific_face(train_test_df = train_test_df, seen_boolean = seen, sampled_face_label = sampled_face_label, seed = seed)

        if print_mode:
            print(' ')
            print('seen: ', seen)
            print('sampled_label: ', sampled_face_label)
            print('edge_cloud_accuracy_gap: ', edge_cloud_accuracy_gap)
            print(' ')

        edge_confidence_vec.append(edge_confidence)
        input_vec.append(embedding_vector)
        edge_cloud_accuracy_gap_vec.append(edge_cloud_accuracy_gap)
        seen_vec.append(seen)
        train_test_membership_vec.append(train_test_membership)

        edge_prediction_vec.append(edge_prediction)
        cloud_prediction_vec.append(cloud_prediction)

        # ground-truth is cloud!!
        true_value_vec.append(cloud_prediction)

        rolling_diff = distance(embedding_vector, past_embedding_vector)
        #embedding_L2_norm = distance(embedding_vector, zero_embedding_vector) 
        embedding_L2_norm = np.mean([np.abs(x) for x in embedding_vector])

        # changed embedding vector
        past_embedding_vector = embedding_vector

        # new colums
        rolling_diff_vec.append(rolling_diff)
        image_name_vec.append(image_name)
        embedding_norm_vec.append(embedding_L2_norm)

    return edge_confidence_vec, edge_prediction_vec, cloud_prediction_vec, true_value_vec, edge_cloud_accuracy_gap_vec, input_vec, seen_vec, rolling_diff_vec, image_name_vec, train_test_membership_vec, embedding_norm_vec

"""
    plot a stochastic episode of the facenet process
"""

def facenet_plot_stochastic_episode(seen_vec = None, edge_confidence_vec = None, edge_cloud_accuracy_gap_vec = None, seed = None, edge_prediction_vec = None, cloud_prediction_vec = None, true_value_vec = None, base_plot_dir = 'plots/', summary_str = 'FaceNet', rolling_diff_vec = None, embedding_norm_vec = None):

    # now draw the overlaid timeseries
    LW = 2.0
    normalized_ts_dict = {}
    normalized_ts_dict['seen distribution'] = {'ts_vector': seen_vec, 'lw': 2.0, 'linestyle': '-'}
    normalized_ts_dict['edge confidence'] = {'ts_vector': edge_confidence_vec, 'lw': 2.5, 'linestyle': '--'}
    normalized_ts_dict['edge-cloud accuracy gap'] = {'ts_vector': edge_cloud_accuracy_gap_vec, 'lw': 3.0, 'linestyle': '-'}
    normalized_ts_dict['embedding (L2 norm)'] = {'ts_vector': embedding_norm_vec, 'lw': 3.0, 'linestyle': '--'}

    plot_file = base_plot_dir + '/' + summary_str + 'overlaid_confidence_ts_' + str(seed) + '.pdf'
    title_str = 'edge confidence utility ' + summary_str 
    overlaid_ts(normalized_ts_dict = normalized_ts_dict, title_str = title_str, plot_file = plot_file, ylabel = None, xlabel = 'time', fontsize = 12, xticks = None, ylim = None, DEFAULT_ALPHA = 1.0, legend_present = True, DEFAULT_MARKERSIZE = 12, delete_yticks = False)

    LW = 2.0
    normalized_ts_dict = {}
    normalized_ts_dict['edge prediction'] = {'ts_vector': edge_prediction_vec, 'lw': 2.0, 'linestyle': '-'}
    normalized_ts_dict['cloud prediction'] = {'ts_vector': cloud_prediction_vec, 'lw': 2.5, 'linestyle': '-'}
    normalized_ts_dict['true value'] = {'ts_vector': true_value_vec, 'lw': 3.0, 'linestyle': '--'}
    normalized_ts_dict['rolling embedding difference (L2 norm)'] = {'ts_vector': rolling_diff_vec, 'lw': 3.0, 'linestyle': '--'}
    #normalized_ts_dict['embedding (L2 norm)'] = {'ts_vector': embedding_norm_vec, 'lw': 3.0, 'linestyle': '--'}

    plot_file = base_plot_dir + '/' + summary_str + 'predictions_overlaid' + str(seed) + '.pdf'
    title_str = 'edge-cloud accuracy gap ' + summary_str
    overlaid_ts(normalized_ts_dict = normalized_ts_dict, title_str = title_str, plot_file = plot_file, ylabel = None, xlabel = 'time', fontsize = 12, xticks = None, ylim = None, DEFAULT_ALPHA = 1.0, legend_present = True, DEFAULT_MARKERSIZE = 12, delete_yticks = False)


if __name__ == '__main__':

    PRINT_MODE = False

    base_pkl_dir = 'MoreFacenet_pkl_data'
    SVM_results_csv = base_pkl_dir + '/SVM_results.csv' 
    
    SVM_results_df = pandas.read_csv(SVM_results_csv)
    print(SVM_results_df.columns)


    # create new plots
    base_plot_dir = 'facenet_ts_plots/'
    remove_and_create_dir(base_plot_dir)

    train_test_vec = ['TRAIN']
    #for trial, seed in enumerate([1,1,3,3]):
    for trial, seed in enumerate([1,2,3]):
            
        for train_test in train_test_vec : 
            edge_confidence_vec, edge_prediction_vec, cloud_prediction_vec, true_value_vec, edge_cloud_accuracy_gap_vec, input_vec, seen_vec, rolling_diff_vec, image_name_vec, train_test_membership_vec, embedding_norm_vec = facenet_stochastic_video(SVM_results_df = SVM_results_df, seed = seed, T = 90, coherence_time = 9, P_SEEN = 0.6, train_test_membership = train_test)

            summary_str = '-'.join(['FaceNet', train_test, str(seed)])

            facenet_plot_stochastic_episode(seen_vec = seen_vec, edge_confidence_vec = edge_confidence_vec, edge_cloud_accuracy_gap_vec = edge_cloud_accuracy_gap_vec, seed = trial, edge_prediction_vec = edge_prediction_vec, cloud_prediction_vec = cloud_prediction_vec, true_value_vec = true_value_vec, base_plot_dir = base_plot_dir, summary_str = summary_str, rolling_diff_vec = rolling_diff_vec, embedding_norm_vec = embedding_norm_vec)



