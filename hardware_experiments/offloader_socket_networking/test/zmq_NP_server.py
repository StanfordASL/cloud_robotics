import sys,os
import zmq
import numpy
import zlib, pickle


base_video_dir = os.environ['CLOUD_ROOT_DIR']
utils_dir = base_video_dir + '/utils/'
sys.path.append(utils_dir)

from zmq_socket_utils import *

# ZeroMQ Context
context = zmq.Context()

# Define the socket using the "Context"
sock = context.socket(zmq.REP)
sock.bind("tcp://127.0.0.1:5678")

# Run a simple "Echo" server
while True:

    A = recv_zipped_pickle(sock)
    #A = recv_array(sock)
    print('got A at server: ', A)

    B_array = numpy.array([1,3,5])


    B = {'frame': 5, 'emb': B_array}

    send_zipped_pickle(sock, B)


