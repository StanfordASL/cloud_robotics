# Author: Sandeep Chinchali

import sys, os
import pandas
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import collections
import argparse

from keras.models import model_from_yaml

from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

from sklearn.externals import joblib

base_video_dir = os.environ['CLOUD_ROOT_DIR']
utils_dir = base_video_dir + '/utils/'
sys.path.append(utils_dir)

from textfile_utils import *
from plotting_utils import *
from DNN_offloader_utils import *

DATA_DIR = base_video_dir + '/data/video_data/'
WORK_DIR = base_video_dir + '/scratch_results/keras_offloader_DNN/'

DEFAULT_train_csv = DATA_DIR + '/total_train_df.csv'

if __name__ == "__main__":

    # arguments
    parser = argparse.ArgumentParser(description='DNNOffloader')
    parser.add_argument('--output-results-dir', type=str, required=False, default = WORK_DIR)
    parser.add_argument('--input-train-csv', type=str, required=False, default = DEFAULT_train_csv)
    parser.add_argument('--predict-var', type=str, required=False, default = 'offload')
    parser.add_argument('--train-features-list', type=str, required=False)

    args = parser.parse_args()

    # where to write results
    WORK_DIR = args.output_results_dir

    # csv of all training data
    train_csv = args.input_train_csv

    # training data
    total_train_df = pandas.read_csv(train_csv)

    # features to train on
    train_features_list = args.train_features_list

    # save_model_dir
    save_model_dir = WORK_DIR + '/offloader_DNN_model/'
    remove_and_create_dir(save_model_dir)
 
    print('columns: ', list(total_train_df))

    # features to train on 
    # ['SVM_confidence', 'embedding_distance', 'face_confidence', 'frame_diff_val', 'numeric_prediction', 'unknown_flag', 'num_detect']
    train_features = list_from_file(train_features_list)

    # what to predict: whether to offload or not
    var_to_predict = [args.predict_var]

    train_df, test_df = train_test_split(total_train_df, test_size=0.2)

    # scale only train data
    scaled_train_features_matrix, train_output_y, feature_scaler, output_scaler = normalize_train_data_ONLY(train_df = train_df, train_features = train_features, var_to_predict = var_to_predict)

    # save scaler
    scaler_filename = save_scaler(scaler = feature_scaler, out_dir = save_model_dir, prefix = 'offload')

    # load scaler
    load_feature_scaler = load_scaler(out_dir = save_model_dir)

    # scale the test data too
    scaled_test_features_matrix, test_output_y = normalize_test_data_ONLY(test_df = test_df, train_features = train_features, var_to_predict = var_to_predict, feature_scaler = load_feature_scaler)

    NN_model_params = {}
    NN_model_params['batch_size'] = 50
    NN_model_params['num_epochs'] = 100
    NN_model_params['num_hidden_units'] = 30
    NN_model_params['early_stop'] = True
    
    # train the NN model
    #############################################
    model, train_results_dict  = train_NN_regression_model(NN_model_params = NN_model_params, scaled_train_features_matrix = scaled_train_features_matrix, train_output_y = train_output_y)
    print(train_results_dict)

    save_keras_model(model = model, save_dir = save_model_dir)

    # test the NN model
    #############################################
    loaded_model = load_keras_model(save_dir = save_model_dir)

    # original
    test_results_dict, predicted_df = test_NN_regression_model(model = loaded_model, scaled_test_features_matrix = scaled_test_features_matrix, test_output_y = test_output_y, test_df = test_df, var_to_predict = var_to_predict)
    print(test_results_dict)

    predictions_csv = WORK_DIR + '/predictions.csv' 
    predicted_df.to_csv(predictions_csv)
