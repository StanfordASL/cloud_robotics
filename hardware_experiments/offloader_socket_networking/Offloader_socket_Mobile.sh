# OFFLOADER CLIENT, sends numpy arrays for facenet embeddings to the cloud server
# invokes offloader DNN to decide what model to use
# can run from a live IP camera or from pre-recorded videos

# where pre-trained facenet models are
DNN_MODEL_DIR=$CLOUD_ROOT_DIR/DNN_models

# code to run the offloader
CODE_DIR=$CLOUD_ROOT_DIR/hardware_experiments/offloader_socket_networking/

# where to put results from runs
RESULTS_DIR=$CLOUD_ROOT_DIR/scratch_results/

# are the facenet DNNs from a large or small version
DNN_PREFIX=SMALL

# where image data resides
DATA_DIR=$CLOUD_ROOT_DIR/data/

PRERECORDED_VIDEO_DIR=$DATA_DIR/prerecorded_videos/

# what video and its annotated output will be named
#PREFIX='csandeep_amine_andrew'
#PREFIX='whole_lab_training'
#PREFIX='james_apoorva_sandeep_1224'
PREFIX=trial_3

PRERECORDED_VIDEO=$PRERECORDED_VIDEO_DIR/${PREFIX}.avi

# write output video, used to display results with bounding boxes
OUT_WRITE_MODE='False'
#OUT_WRITE_MODE='True'

# whether to write the offloading decisions to a file to compute the accuracy later on
FILE_LOG_OUT='False'

ROBOT_SVM_RESULTS_DIR=$DNN_MODEL_DIR/${DNN_PREFIX}_EDGE_SVM/
CLOUD_SVM_RESULTS_DIR=$DNN_MODEL_DIR/${DNN_PREFIX}_CLOUD_SVM/

# use live nvidia camera or not, otherwise use pre-recorded video
#CAMERA_MODE='True'
CAMERA_MODE='False'

#HOST_IP='127.0.0.1'
HOST_IP='192.168.1.101'
PORT=8000

OFFLOAD_DNN_MODEL_DIR=$DNN_MODEL_DIR/offloader_DNN_logic/keras_offloader_DNN/

# start the mobile client which offloads facenet computations over python sockets
# can test with loopback and then send via WiFi over a live network
# main args: needs the face detector, the openface model, SVMs for robot and cloud, and the NN offloading logic

python3 -i $CODE_DIR/socket_offloader_video.py  --detector $DNN_MODEL_DIR/face_detection_model \
    --embedding-model $DNN_MODEL_DIR/openface_nn4.small2.v1.t7 \
    --robot_recognizer $ROBOT_SVM_RESULTS_DIR/recognizer.pickle \
    --robot_le $ROBOT_SVM_RESULTS_DIR/le.pickle \
    --cloud_recognizer $CLOUD_SVM_RESULTS_DIR/recognizer.pickle \
    --cloud_le $CLOUD_SVM_RESULTS_DIR/le.pickle \
    -cam $CAMERA_MODE -outprefix $PREFIX -out $OUT_WRITE_MODE --prerecorded_video $PRERECORDED_VIDEO \
	--host $HOST_IP --port $PORT --OFFLOAD_DNN_MODEL_DIR $OFFLOAD_DNN_MODEL_DIR \
	--file_log_out ${FILE_LOG_OUT} --base_run_results_dir $RESULTS_DIR
