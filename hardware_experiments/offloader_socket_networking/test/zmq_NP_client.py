import zmq
import sys,os
import numpy
import zlib, pickle

base_video_dir = os.environ['CLOUD_ROOT_DIR']
utils_dir = base_video_dir + '/utils/'
sys.path.append(utils_dir)

from zmq_socket_utils import *

# ZeroMQ Context
context = zmq.Context()

# Define the socket using the "Context"
sock = context.socket(zmq.REQ)
sock.connect("tcp://127.0.0.1:5678")


A = numpy.array([1,3,4,5])

while True:
    send_zipped_pickle(sock, A)


    B = recv_zipped_pickle(sock)
    print('received from server at client B: ', B)


#send_array(sock, A)

# Send a "message" using the socket
#sock.send(b" ".join(sys.argv[1:]))
#print(sock.recv())
