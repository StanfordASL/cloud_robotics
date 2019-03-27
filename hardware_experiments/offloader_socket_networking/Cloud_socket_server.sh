# are the facenet DNNs from a large or small version
DNN_PREFIX=SMALL

# where pre-trained facenet models are
DNN_MODEL_DIR=$CLOUD_ROOT_DIR/DNN_models

CLOUD_SVM_RESULTS_DIR=${DNN_MODEL_DIR}/${DNN_PREFIX}_CLOUD_SVM/

echo $CLOUD_SVM_RESULTS_DIR

HOST=192.168.1.101
PORT=8000

python3 video_central_facenet_socket_server.py  \
    --cloud_recognizer $CLOUD_SVM_RESULTS_DIR/recognizer.pickle \
    --cloud_le $CLOUD_SVM_RESULTS_DIR/le.pickle \
    --host $HOST --port $PORT
