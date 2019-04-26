# input data from videos in csv format to train on
INPUT_TRAIN_CSV=$CLOUD_ROOT_DIR/data/video_data/total_train_df.csv

# where to store the offloader model in KERAS
OUTPUT_RESULTS_DIR=$CLOUD_ROOT_DIR/scratch_results/keras_offloader_DNN/

# what is the column name we want to predict: whether to offload or not
PREDICT_VAR='offload'

# what input features do we use to decide if we should offload?
TRAIN_FEATURES_LIST=$PWD/offloader_DNN_configs/offloader_DNN_input_features.txt

# train the offloader DNN
python3 train_supervised_learning_offloader.py --output-results-dir $OUTPUT_RESULTS_DIR --input-train-csv $INPUT_TRAIN_CSV --predict-var $PREDICT_VAR --train-features-list $TRAIN_FEATURES_LIST
