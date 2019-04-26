# Author: Sandeep Chinchali

import sys, os
import pandas
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import collections
from collections import OrderedDict

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

# run experiments in parallel
early_stopping = EarlyStopping(monitor='val_loss', patience=2)

# GLOBAL CONSTANTS
##########################################
prediction_to_numeric_dict = OrderedDict()
prediction_to_numeric_dict['unknown'] = -1
prediction_to_numeric_dict['apoorva'] = 0
prediction_to_numeric_dict['james'] = 1
prediction_to_numeric_dict['sandeep'] = 2
prediction_to_numeric_dict['joe'] = 3
prediction_to_numeric_dict['karen'] = 4
prediction_to_numeric_dict['boris'] = 5
prediction_to_numeric_dict['amine'] = 6
prediction_to_numeric_dict['andrew'] = 7
prediction_to_numeric_dict['adrian'] = 8
prediction_to_numeric_dict['trisha'] = 9
prediction_to_numeric_dict['abi'] = 10

##########################################
"""
error metrics
"""

def calc_error_metrics(true_values = None, predicted_values = None):

    r2 = r2_score(true_values, predicted_values)

    mse = np.mean((true_values - predicted_values)**2)

    rmse = mse**(0.5)

    # percentage wrong

    predicted_binary = [int(pred > 0.5) for pred in predicted_values]
    true_binary = [true_y for true_y in true_values]

    mpe = np.mean([int(pred != true_y) for pred, true_y in zip(predicted_binary, true_binary)]) 

    return r2, rmse, mpe




"""
"""

def build_single_layer_model(data_dim, num_hidden_units):
    """
    Compile a simple NN for regression
    """ 
    model = Sequential()

    model.add(Dense(num_hidden_units, input_dim=data_dim, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_hidden_units, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    #model.add(Dense(num_hidden_units, kernel_initializer='normal', activation='relu'))
    #model.add(Dropout(0.2))
    
    # regression
    #model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    #model.compile(loss='mean_squared_error', optimizer='adam')

    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    print("Model parameters: {}".format(model.count_params()))
    return model

def normalize_train_test_data_for_NN(train_df = None, test_df = None, train_features = None, var_to_predict = None):

    # scale the data
    #############################################
    train_features_matrix = train_df[train_features].as_matrix()
    train_output_y = train_df[var_to_predict].as_matrix()
   
    test_features_matrix = test_df[train_features].as_matrix()
    test_output_y = test_df[var_to_predict].as_matrix()

    print('fit train feature scaler')
    feature_scaler = RobustScaler().fit(train_features_matrix)

    print('fit output scaler')
    output_scaler = RobustScaler().fit(train_output_y)

    scaled_train_features_matrix = feature_scaler.transform(train_features_matrix)
    scaled_train_output_y_matrix = output_scaler.transform(train_output_y)
    
    scaled_test_features_matrix = feature_scaler.transform(test_features_matrix)
    scaled_test_output_y_matrix = output_scaler.transform(test_output_y)

    return scaled_train_features_matrix, train_output_y, scaled_test_features_matrix, test_output_y, feature_scaler, output_scaler

#####
# normalize train data only
#####
def normalize_train_data_ONLY(train_df = None, train_features = None, var_to_predict = None):

    # scale the data
    #############################################
    train_features_matrix = train_df[train_features].as_matrix()
    train_output_y = train_df[var_to_predict].as_matrix()
   
    print('fit train feature scaler')
    feature_scaler = RobustScaler().fit(train_features_matrix)

    print('fit output scaler')
    output_scaler = RobustScaler().fit(train_output_y)

    scaled_train_features_matrix = feature_scaler.transform(train_features_matrix)
    scaled_train_output_y_matrix = output_scaler.transform(train_output_y)
    
    return scaled_train_features_matrix, train_output_y, feature_scaler, output_scaler

##################################

def normalize_test_data_ONLY(test_df = None, train_features = None, var_to_predict = None, feature_scaler = None):
    # scale the data
    #############################################
    test_features_matrix = test_df[train_features].as_matrix()
    test_output_y = test_df[var_to_predict].as_matrix()
   
    scaled_test_features_matrix = feature_scaler.transform(test_features_matrix)
    
    return scaled_test_features_matrix, test_output_y

##################################
def train_NN_regression_model(NN_model_params = None, scaled_train_features_matrix = None, train_output_y = None, model_save_params_dict = None, validation_frac = 0.2):
    # train the NN model
    #############################################
    data_dim = scaled_train_features_matrix.shape[1]
    
    model = build_single_layer_model(data_dim, NN_model_params['num_hidden_units'])

    if NN_model_params['early_stop']:
        model.fit(
            scaled_train_features_matrix,
            train_output_y,
            batch_size=NN_model_params['batch_size'],
            epochs=NN_model_params['num_epochs'],
            validation_split=validation_frac,
            callbacks = [early_stopping])

    else:
        model.fit(
            scaled_train_features_matrix,
            train_output_y,
            batch_size=NN_model_params['batch_size'],
            epochs=NN_model_params['num_epochs'])

    train_predictions = model.predict(scaled_train_features_matrix)
    train_r2, train_rmse, train_mpe = calc_error_metrics(true_values = train_output_y, predicted_values = train_predictions[:,0])
    
    train_results_dict = {}
    train_results_dict = {'train_r2': train_r2, 'train_rmse': train_rmse, 'train_mpe': train_mpe, 'train_num_pts': len(train_output_y)}
   
    # save the pkl model
    #######################################
    # eventually save keras NN model
    return model, train_results_dict 

##################################
def test_NN_regression_model(model = None, scaled_test_features_matrix = None, test_output_y = None, test_df = None, var_to_predict = None, date_var = None):

    test_predictions = model.predict(scaled_test_features_matrix)

    print('Test MSE: {}'.format(mean_squared_error(test_output_y, test_predictions)))
    test_r2, test_rmse, test_mpe = calc_error_metrics(true_values = test_output_y, predicted_values = test_predictions[:,0])

    predicted_df = test_df.copy()
    predicted_df['test_predictions'] = test_predictions

    test_results_dict = {}
    test_results_dict = {'test_r2': test_r2, 'test_rmse': test_rmse, 'test_mpe': test_mpe, 'test_num_pts': test_df.shape[0]}
    
    return test_results_dict, predicted_df

# save keras model
##################################

def save_keras_model(model = None, save_dir = None, prefix = 'offload'):
    # serialize model to YAML
    model_yaml = model.to_yaml()
    
    model_yaml_name = save_dir + '/model.' + str(prefix) + '.yaml'      
    model_weights_name = save_dir + '/weights.' + str(prefix) + '.h5'   

    with open(model_yaml_name, "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    model.save_weights(model_weights_name)
    print("Saved model: ", model_yaml_name, model_weights_name)

    return model_yaml_name, model_weights_name

# load saved keras model
##################################

def load_keras_model(save_dir = None, prefix = 'offload'):
    model_yaml_name = save_dir + '/model.' + str(prefix) + '.yaml'      
    model_weights_name = save_dir + '/weights.' + str(prefix) + '.h5'   

    # load YAML and create model
    yaml_file = open(model_yaml_name, 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()

    loaded_model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    loaded_model.load_weights(model_weights_name)
    print("Loaded model: ", model_yaml_name, model_weights_name)

    return loaded_model

def save_scaler(scaler = None, out_dir = None, prefix = None):
    scaler_filename = out_dir + "/feature_scaler.save"
    joblib.dump(scaler, scaler_filename) 
    return scaler_filename

def load_scaler(out_dir = None):
    scaler_filename = out_dir + "/feature_scaler.save"
    scaler = joblib.load(scaler_filename)   
    print('load scaler: ', scaler_filename) 
    return scaler

