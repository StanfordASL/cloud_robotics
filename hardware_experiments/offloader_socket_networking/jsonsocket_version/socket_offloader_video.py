# mobile offloading client

# this is only on the jetson for a conflict with ROS's openCV
import sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    print('no ROS')
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import sys,os
from jsonsocket import Client, Server

# generic utils and root dir
base_video_dir = os.environ['CLOUD_ROOT_DIR']
utils_dir = base_video_dir + '/utils/'
sys.path.append(utils_dir)

# utils for the DNN offloader
offload_utils_dir = base_video_dir + '/hardware_experiments/offloader_implementation/'
sys.path.append(offload_utils_dir)

# from main utils dir
from textfile_utils import *
from calculation_utils import *

# from offload utils
from utils_offload import *
from keras_offload_DNN import *

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True,
        help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
        help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--robot_recognizer", required=True,
        help="path to model trained to recognize faces")
ap.add_argument("-l", "--robot_le", required=True,
        help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
ap.add_argument("-cam", "--camera_mode", type=str, default='False',
        help="whether to use onboard camera")
ap.add_argument("-prerec", "--prerecorded_video", type=str, default=None,
        help="pre-recorded video")
ap.add_argument("-outprefix", "--prefix_outvideo", type=str, default='detectOutVideo',
        help="out_video_prefix")
ap.add_argument("-out", "--out_video_write_mode", type=str, default=False,
        help="pre-recorded video")
ap.add_argument("-rc", "--cloud_recognizer", required=True,
        help="path to model trained to recognize faces")
ap.add_argument("-lc", "--cloud_le", required=True,
        help="path to label encoder")
ap.add_argument("-fileLogOut", "--file_log_out", type=str, default='False',
        help="pre-recorded video")
ap.add_argument("-hst", "--host", type=str, default='localhost',
        help="IP for host")
ap.add_argument("-p", "--port", type=int, default=8000,
        help="port for host")
ap.add_argument("-NN", "--OFFLOAD_DNN_MODEL_DIR", type=str, default=None, help='where offloader DNN is stored')
ap.add_argument("-RES", "--base_run_results_dir", type=str, default=None, help='where results should go')

args = vars(ap.parse_args())


# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
        "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# load the actual face recognition model along with the label encoder
# this is a pretrained SVM to recognize known faces
# this is the EDGE MODEL
robot_recognizer = pickle.loads(open(args["robot_recognizer"], "rb").read())
robot_le = pickle.loads(open(args["robot_le"], "rb").read())

# analagous CLOUD MODEL
cloud_recognizer = pickle.loads(open(args["cloud_recognizer"], "rb").read())
cloud_le = pickle.loads(open(args["cloud_le"], "rb").read())

# OFFLOADER NEURAL NET IS PRE-TRAINED: 
######################################
# load the model
OFFLOAD_DNN_MODEL_DIR = args['OFFLOAD_DNN_MODEL_DIR']
NN_model = load_keras_model(save_dir = OFFLOAD_DNN_MODEL_DIR, prefix = 'offload')

# load feature scaler for offloader DNN
feature_scaler = load_scaler(out_dir = OFFLOAD_DNN_MODEL_DIR)

# features the offloader NN expects as input
train_features = ['SVM_confidence', 'embedding_distance', 'face_confidence', 'frame_diff_val', 'numeric_prediction', 'unknown', 'num_detect']
######################################

# START THE MOBILE CLIENT
#########################################
host = args['host']
port = int(args['port'])

# Client code:
client = Client()

# setup for making the output video
#########################################
frame_width = 450
frame_height = 600

prefix = args['prefix_outvideo']

# do we write an output video?
if args['out_video_write_mode'] == 'True':
    out_video_write_mode = True
else:
    out_video_write_mode = False

# do we write output files?
if args['file_log_out'] == 'True':
    file_log_out = True
else:
    file_log_out = False

# only if we save results of offloading or video, do we make an output dir
if (out_video_write_mode or file_log_out):

    # where to save output video
    output_results_dir = args['base_run_results_dir'] + '/' +  args["prefix_outvideo"] + '/'
    remove_and_create_dir(output_results_dir) 

    # where to store specific frames
    specific_frame_dir = output_results_dir + '/selected_frame_images/'
    remove_and_create_dir(specific_frame_dir)


# do we write an output video?
if out_video_write_mode:
        # Define the codec and create VideoWriter object
        # this is for saving video
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        out_file = output_results_dir + '/offloader_SVM_output_' + args['prefix_outvideo'] + '.avi'
        out = cv2.VideoWriter(out_file, fourcc, 20, (frame_height, frame_width))

# do we write output files about offloading statistics made?
if file_log_out:

        # a text file of all the offloading decisions
        # create this in all cases
        out_csv = output_results_dir + '/offloader_SVM_output_' + args['prefix_outvideo'] + '.txt'
        out_file = open(out_csv, 'w')
        header_str = '\t'.join(['frame_number', 'SVM_confidence', 'prediction', 'face_confidence', 'embedding_distance', 'frame_diff_val', 'unknown', 'numeric_prediction', 'offload', 'offload_prob', 'cloud_name', 'cloud_SVM_proba', 'box_area'])
        out_file.write(header_str + '\n')

# initialize the video stream, then allow the camera sensor to warm up
# use the onboard NVIDIA JETSON CAMERA
if args['camera_mode'] == 'True':
        print("[INFO] LIVE STREAM FROM CAMERA ")

        # can change this for a local IP camera
        gst_str = "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)I420, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"
        out_file_prefix = 'nvidia'

        vs = cv2.VideoCapture(gst_str)
        time.sleep(2.0)

# stream video from a prerecorded file
else:
        print("[INFO] prerecorded_str ", args['prerecorded_video'])
        vs = cv2.VideoCapture(args['prerecorded_video'])
        time.sleep(2.0)


# KEY STATISTICS ON HOW WE PLAY THE VIDEO
###################################################################
# counter of video frames
frame_number = 0

# wait betwen frames, play with this if going too slowly
waitkey_duration = 1

# start the FPS throughput estimator
fps = FPS().start()

# run the DNN every FRAME_POLL frames
FRAME_POLL = 1
# how often to write offloading results to a txt file for plotting
WRITE_INTERVAL = 5
# how often to print results
PRINT_INTERVAL = 10
# whether to draw annotations on images
DRAW_BOXES_MODE = True
# confidence for a true detection
SVM_THRESHOLD = 30
# whether to display live video in a monitoring window
VIDEO_DISPLAY_MODE = True

# just a proxy for how much face embeddings change across frames
# embedding distances dict
# per name, the past embedding we had to figure out if face is changing much
embedding_dict = {}
EMBEDDING_DIMENSION =  128
# loop thru possible names
for name in robot_le.classes_:
        embedding_dict[name] = np.zeros(EMBEDDING_DIMENSION)

# keep a tally of how frames are changing
past_frame = None

# colors to display on openCV image frames
RED_COLOR = (0, 0, 255)
BLUE_COLOR = (128, 0, 0)
YELLOW_COLOR = (0, 255, 255)

# MAIN WHILE LOOP TO PARSE THRU THE VIDEO
###################################################################
# loop over frames from the video file stream
while vs.isOpened():
        # grab the frame from the threaded video stream
        ret, frame = vs.read()

        # resize the frame to have a width of 600 pixels (while
        # maintaining the aspect ratio), and then grab the image
        # dimensions
        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]

        # frame diff is a differencing frame, use to plot diffs
        if frame_number > 0:
                frame_diff = cv2.absdiff(frame, past_frame)     
        else:
                frame_diff = frame

        # quantify how much frames are changing
        frame_diff_val = np.sum(frame_diff)/(h * w * 3)

        # update past frame
        past_frame = frame.copy()

        # run the DNN only every FRAME_POLL frames
        if frame_number % FRAME_POLL == 0:

                # construct a blob from the image
                imageBlob = cv2.dnn.blobFromImage(
                        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                        (104.0, 177.0, 123.0), swapRB=False, crop=False)

                # apply OpenCV's deep learning-based face detector to localize
                # faces in the input image
                detector.setInput(imageBlob)
                detections = detector.forward()

                # loop over the detections
                num_detect = 0
                at_least_one_offload = False
                for i in range(0, detections.shape[2]):
                        # extract the confidence (i.e., probability) associated with
                        # the prediction
                        confidence = detections[0, 0, i, 2]

                        # filter out weak detections
                        if confidence > args["confidence"]:
                                # compute the (x, y)-coordinates of the bounding box for
                                # the face
                                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                                (startX, startY, endX, endY) = box.astype("int")

                                # extract the face ROI
                                face = frame[startY:endY, startX:endX]
                                (fH, fW) = face.shape[:2]
                                
                                # ensure the face width and height are sufficiently large
                                if fW < 20 or fH < 20:
                                        continue
                                box_area = fW * fH

                                num_detect += 1
                                # construct a blob for the face ROI, then pass the blob
                                # through our face embedding model to obtain the 128-d
                                # quantification of the face
                                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                        (96, 96), (0, 0, 0), swapRB=True, crop=False)
                                embedder.setInput(faceBlob)
                                vec = embedder.forward()

                                # perform classification to recognize the face
                                # run the SVM to recognize a face from its embedding
                                preds = robot_recognizer.predict_proba(vec)[0]
                                j = np.argmax(preds)
                                proba = preds[j]
                                name = robot_le.classes_[j]

                                # per person, store how the embedding frames are changing
                                past_embedding_vec = embedding_dict[name]
                                embedding_distance = distance(past_embedding_vec, vec[0])
                                embedding_dict[name] = vec[0]

                                # store SVM confidence for face
                                SVM_confidence = proba * 100

                                # offload logic
                                # here we are offloading SPECIFIC FACES!
                                ########################################

                                numeric_prediction = get_numeric_prediction(name = name)

                                unknown = int(name == 'unknown')

                                # feature values, input to NN
                                face_confidence = confidence
                                feature_values = [SVM_confidence, embedding_distance, face_confidence, frame_diff_val, numeric_prediction, unknown, num_detect]

                                # single training input to NN
                                row_dict = assemble_row_df(train_features = train_features, feature_values = feature_values)
                            
                                # call the offloader DNN    
                                offload_decision, offload_prob = predict_offloader(row_data_pandas = row_dict, train_features = train_features, feature_scaler = feature_scaler, NN_model = NN_model)


                                # just a test that periodically offloads
                                ########################################
                                #if frame_number % 20 == 0:
                                #    offload_decision = True
                                #    offload_prob = 1.0
                                #else:
                                #    offload_decision = False
                                #    offload_prob = 0.0

                                # ONLY IF WE OFFLOAD, query the cloud model with the embedding
                                if offload_decision:
                                    # whether we have offloaded once for this frame
                                    at_least_one_offload = True

                                    # embedding vector to send to cloud server over a socket
                                    embedding_vec_to_send_dict = {'frame': frame_number, 'emb': [str(x) for x in vec[0]]}

                                    # connect to server and send
                                    client.connect(host, port).send(embedding_vec_to_send_dict)

                                    # recieve the results back
                                    cloud_response_dict = client.recv()
                                   
                                    # unpack the label from the cloud
                                    cloud_name = cloud_response_dict['cloud_name']
                                    cloud_SVM_proba = cloud_response_dict['cloud_SVM_proba']
                                    cloud_numeric_prediction = cloud_response_dict['cloud_numeric_prediction']

                                    print('frame: ', frame_number, 'RESPONSE CLOUD NAME: ', cloud_name, 'RESPONSE SVM PROBA: ', cloud_SVM_proba, 'CLOUD PRED: ', cloud_numeric_prediction)
                                else:
                                    # this is PURELY for training a model
                                    cloud_name, cloud_SVM_proba = query_facenet_model_SVM(vec = vec, recognizer = cloud_recognizer, le = cloud_le)
                                    cloud_numeric_prediction = get_numeric_prediction(name = cloud_name)

                                # print status to screen every so often
                                if frame_number % PRINT_INTERVAL == 0:
                                        print(' ')              
                                        print('frame_number: ', frame_number, 'name: ', name, 'SVM_confidence: ', round(SVM_confidence, 3), 'embedding_dist: ', round(embedding_distance, 4), 'fram_diff:', frame_diff.shape, 'frame_diff_val: ', frame_diff_val, 'num_detect: ', num_detect)
                                        print('offload_decision: ', offload_decision, 'prob: ', offload_prob)
                                        print(' ')              

                                # draw a box around faces       
                                if (DRAW_BOXES_MODE and (SVM_confidence > SVM_THRESHOLD)):

                                        # draw the bounding box of the face along with the
                                        # associated probability
                                        y = startY - 10 if startY - 10 > 10 else startY + 10

                                        text1y = y = startY - 10 if startY - 10 > 10 else startY + 20
                                        text2y = y = endY + 10 if endY + 10 > 10 else endY - 10
                                        SCALE = 0.75
                                        
                                        for type_index, frame_type in enumerate([frame, frame_diff]):
                                            if type_index == 0:
                                                numeric_mode = False
                                            else:
                                                numeric_mode = True

                                            # for the difference frame 
                                            if offload_decision == 1:
                                                # original
                                                # text = "{}: {:.2f}%".format('Cloud, ' + display_name, proba * 100)
                                                # COLOR = YELLOW_COLOR
                                                
                                                cloud_display_name = get_display_name(name = cloud_name, NUMERIC_MODE = numeric_mode, numeric_prediction = cloud_numeric_prediction)
                                                robot_display_name = get_display_name(name = name, NUMERIC_MODE = numeric_mode, numeric_prediction = numeric_prediction)

                                                cloud_str = "{}: {:.2f}%".format('Cloud, ' + cloud_display_name, cloud_SVM_proba)
                                                robot_str = "{}: {:.2f}%".format('Robot, ' + robot_display_name, proba * 100)
                                                FRAME_COLOR = YELLOW_COLOR
                                                
                                                # put box
                                                cv2.rectangle(frame_type, (startX, startY), (endX, endY),
                                                    FRAME_COLOR, 2)
                                                
                                                # robot str
                                                cv2.putText(frame_type, robot_str, (startX, text1y),
                                                    cv2.FONT_HERSHEY_SIMPLEX, SCALE, RED_COLOR, 2)

                                                # cloud str
                                                cv2.putText(frame_type, cloud_str, (startX, text2y),
                                                    cv2.FONT_HERSHEY_SIMPLEX, SCALE, YELLOW_COLOR, 2)

                                            else: 
                                                FRAME_COLOR = RED_COLOR
                                                
                                                robot_display_name = get_display_name(name = name, NUMERIC_MODE = numeric_mode, numeric_prediction = numeric_prediction)
                                                robot_str = "{}: {:.2f}%".format('Robot, ' + robot_display_name, proba * 100)
                                                # put box
                                                cv2.rectangle(frame_type, (startX, startY), (endX, endY),
                                                    FRAME_COLOR, 2)
                                                
                                                # robot str
                                                cv2.putText(frame_type, robot_str, (startX, text1y), cv2.FONT_HERSHEY_SIMPLEX, SCALE, RED_COLOR, 2)

                                # write every so often to a results file
                                if (frame_number % WRITE_INTERVAL == 0) and (file_log_out):

                                    # THE COLUMNS TO LOG
                                    # header_str = '\t'.join(['frame_number', 'SVM_confidence', 'prediction', 'face_confidence', 'embedding_distance', 'frame_diff_val', 'unknown', 'numeric_prediction', 'offload', 'offload_prob'])
                                    out_str = '\t'.join([str(frame_number), str(proba*100), str(name), str(confidence), str(embedding_distance), str(frame_diff_val), str(unknown), str(numeric_prediction), str(offload_decision), str(offload_prob), str(cloud_name), str(cloud_SVM_proba), str(box_area)])
                                    out_file.write(out_str + '\n')
                                    out_file.flush()


        # update the FPS counter
        fps.update()

        # update frame number
        frame_number += 1
        
        # show the output frame
        if VIDEO_DISPLAY_MODE:
            cv2.imshow("Video", frame_diff)
            #cv2.imshow("Video", frame)

        # save certain frames to an output directory for the paper
        if (out_video_write_mode or file_log_out):
            if ( (at_least_one_offload) and (num_detect >= 2) and (frame_diff_val > 7.0) ):
                    # draw frame difference
                    save_frame = specific_frame_dir + prefix + 'diff-Frame-' + str(frame_number) + '.jpg'
                    cv2.imwrite(save_frame, frame_diff)

                    # draw actual frame
                    save_frame = specific_frame_dir + prefix + 'Frame-' + str(frame_number) + '.jpg'
                    cv2.imwrite(save_frame, frame)

        key = cv2.waitKey(waitkey_duration) & 0xFF

        # write frame to output video
        if out_video_write_mode:
                #out.write(frame)
                out.write(frame_diff)

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
                out_file.close()
                break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
if out_video_write_mode:
        out.release()

if file_log_out:
   out_file.close() 


# close the json client
client.close()

cv2.destroyAllWindows()
vs.release()
