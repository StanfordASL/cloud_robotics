import sys,os
from scipy.misc import imresize
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2


base_video_dir = os.environ['CLOUD_ROOT_DIR']
utils_dir = base_video_dir + '/utils/'
sys.path.append(utils_dir)

from numpysocket import NumpySocket

# For local testing
host_ip = 'localhost'

# on a WiFi network, determine this using ifconfig
host_ip   = '192.168.1.101'


# address of the camera for nvidia jetson TX2
gst_str = "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)I420, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"

cap = cv2.VideoCapture(gst_str)
npSocket = NumpySocket()
npSocket.startServer(host_ip, 9999)

print('here')

# Read until video is completed
print(cap.isOpened())


frame_number = 0
SEND_INTERVAL = 5
while(cap.isOpened()):
    ret, frame = cap.read()
    #print('ret: ', ret)
    ref_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_resize = imresize(ref_frame, .5)
    if ret is True:

        if (frame_number % SEND_INTERVAL == 0):
            print('sending')
            npSocket.sendNumpy(frame_resize)
    else:
        pass
    frame_number += 1


# When everything done, release the video capture object
npSocket.endServer()
cap.release()
