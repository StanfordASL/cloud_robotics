#!/usr/bin/env python

import sys,os
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
import argparse
import pickle
from jsonsocket import Client, Server
import time

from socket import error as SocketError

base_video_dir = os.environ['CLOUD_ROOT_DIR']
utils_dir = base_video_dir + '/utils/'
sys.path.append(utils_dir)

offload_utils_dir = base_video_dir + '/hardware_experiments/offloader_implementation/'
sys.path.append(offload_utils_dir)

from utils_offload import *

if __name__ == '__main__':


    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-rc", "--cloud_recognizer", required=True,
            help="path to model trained to recognize faces")
    ap.add_argument("-lc", "--cloud_le", required=True,
            help="path to label encoder")
    ap.add_argument("-hst", "--host", type=str, default='localhost',
            help="IP for host")
    ap.add_argument("-p", "--port", type=str, default=8000,
            help="port for host")

    print_mode = True

    args = vars(ap.parse_args())

    GLOBAL_CLOUD_RECOGNIZER = pickle.loads(open(args["cloud_recognizer"], "rb").read())
    GLOBAL_CLOUD_LE = pickle.loads(open(args["cloud_le"], "rb").read())

    host = args['host']
    port = args['port']

    # Server code:
    server = Server(host, port)

    # Read until video is completed
    print('ready to accept connections at server')
    while(True):

        # Capture frame-by-frame
        server.accept()
        received_query = server.recv()

        frame = received_query['frame']
        embedding_vec = np.array([float(x) for x in received_query['emb']]).reshape(1,-1)

        if print_mode:
            print(' ')
            print('server query at frame: ', frame)

        cloud_name, cloud_SVM_proba = query_facenet_model_SVM(vec = embedding_vec, recognizer = GLOBAL_CLOUD_RECOGNIZER, le = GLOBAL_CLOUD_LE)
        cloud_numeric_prediction = get_numeric_prediction(name = cloud_name)
        cloud_response_dict = {'cloud_name': cloud_name, 'cloud_SVM_proba': cloud_SVM_proba, 'cloud_numeric_prediction': cloud_numeric_prediction}

        server.send(cloud_response_dict)

        if print_mode:
            print('sent server response: ', cloud_response_dict)
            print(' ')

        #time.sleep(0.01)
        ## Press Q on keyboard to  exit
        #if 0xFF == ord('q'):
        #    break


server.close()
