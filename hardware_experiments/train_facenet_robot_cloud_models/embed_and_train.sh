# is this a LARGE or SMALL facenet model?
# LARGE: nn4.v2.t7
# SMALL: openface_nn4.small2.v1.t7
DNN_PREFIX=SMALL

# where pre-trained facenet models are
DNN_MODEL_DIR=$CLOUD_ROOT_DIR/DNN_models

# choose the small openface model
FACENET_DNN=$DNN_MODEL_DIR/openface_nn4.small2.v1.t7

# uncomment if we want the LARGE one
#FACENET_DNN=$DNN_MODEL_DIR/nn4.v2.t7

# code to train SVM: this directory
CODE_DIR=$CLOUD_ROOT_DIR/hardware_experiments/train_facenet_robot_cloud_models/

# where to put results
RESULTS_DIR=$CLOUD_ROOT_DIR/scratch_results/

# where image data resides
DATA_DIR=$CLOUD_ROOT_DIR/data/

# train an EDGE AND CLOUD model, where EDGE = ROBOT
for PREFIX in EDGE CLOUD

do
    echo $PREFIX

    DATASET=${DATA_DIR}/${PREFIX}_DATASET
    echo $DATASET

	# where the svm models will go
    SVM_RESULTS_DIR=${DNN_MODEL_DIR}/${DNN_PREFIX}_${PREFIX}_SVM/
    rm -rf $SVM_RESULTS_DIR
    mkdir -p $SVM_RESULTS_DIR

    echo 'start embeddings'

    python3 $CODE_DIR/extract_embeddings.py --dataset $DATASET \
        --embeddings $SVM_RESULTS_DIR/embeddings.pickle \
        --detector $DNN_MODEL_DIR/face_detection_model \
        --embedding-model $FACENET_DNN

    echo 'training SVM'

    python3 $CODE_DIR/train_SVM_facenet_model.py --embeddings $SVM_RESULTS_DIR/embeddings.pickle \
        --recognizer $SVM_RESULTS_DIR/recognizer.pickle \
        --le $SVM_RESULTS_DIR/le.pickle
done
