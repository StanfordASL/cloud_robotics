# ## From https://stackoverflow.com/questions/30988033/sending-live-video-frame-over-network-in-python-opencv
# import socket


import sys,os
base_video_dir = os.environ['CLOUD_ROOT_DIR']
utils_dir = base_video_dir + '/utils/'
sys.path.append(utils_dir)

from numpysocket import NumpySocket
import cv2

npSocket = NumpySocket()
npSocket.startClient(9999)

# Read until video is completed
while(True):
    # Capture frame-by-frame
    frame = npSocket.recieveNumpy()
    cv2.imshow('Frame', frame)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

npSocket.endServer()
print("Closing")
