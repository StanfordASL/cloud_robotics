# import the necessary packages

import sys,os
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import sys,os
import pandas
from collections import OrderedDict

import warnings
warnings.filterwarnings("ignore")


base_video_dir = os.environ['CLOUD_ROOT_DIR']
utils_dir = base_video_dir + '/utils/'
sys.path.append(utils_dir)

from textfile_utils import *
#from plotting_utils import *
#from keras_offload_DNN import *

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

def get_numeric_prediction(name = None):
    return prediction_to_numeric_dict[name]

def get_display_name(name = None, numeric_prediction = None, NUMERIC_MODE = True):
    if NUMERIC_MODE:
        if name != 'unknown':
            display_name = 'ID-' + str(numeric_prediction)
        else:
            display_name = 'unknown'
    else:
        display_name = name

    return display_name

def assemble_row_df(train_features = None, feature_values = None):

    row_dict = OrderedDict()
    for k,v in zip(train_features, feature_values):
        row_dict[k] = [v]

    return row_dict



def predict_offloader(row_data_pandas = None, train_features = None, feature_scaler = None, NN_model = None, print_mode = False):

    row_df = pandas.DataFrame(row_data_pandas)
    # scale the prediction
    train_features_matrix = row_df[train_features].as_matrix()
    #train_features_matrix = row_df[train_features].values()


    scaled_input_features = feature_scaler.transform(train_features_matrix)


    # predicted decision
    predict_offload_decision = NN_model.predict(scaled_input_features)[0][0]


    binary_offload_decision = int(predict_offload_decision > 0.50)

    if print_mode:
        print('row_df')
        print(' ')
        print(row_df)
        
        print('train_features_matrix')
        print(train_features_matrix)
        print(' ')

        print('scaled')
        print(scaled_input_features)
        print(' ')
        
        print('binary')
        print(binary_offload_decision)
        print(' ')

        print('predict')
        print(predict_offload_decision)
        print(' ')

    return binary_offload_decision, predict_offload_decision

def query_facenet_model_SVM(vec = None, recognizer = None, le = None):
    preds = recognizer.predict_proba(vec)[0]
    j = np.argmax(preds)
    proba = preds[j]
    name = le.classes_[j]

    SVM_proba = proba * 100
    return name, SVM_proba 

def generate_embedding(faceBlob = None, embedder = None):
    embedder.setInput(faceBlob)
    vec = embedder.forward()
    return vec 
