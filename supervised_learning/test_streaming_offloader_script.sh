# where to store the offloader model in KERAS
OUTPUT_RESULTS_DIR=$CLOUD_ROOT_DIR/scratch_results/keras_offloader_DNN/

# what is the column name we want to predict: whether to offload or not
PREDICT_VAR='offload'

# what input features do we use to decide if we should offload?
TRAIN_FEATURES_LIST=$PWD/offloader_DNN_configs/offloader_DNN_input_features.txt

CONFIGS_DIR=$PWD/offloader_DNN_configs/

TRAIN_TS_CSV_LIST=$CONFIGS_DIR/train_ts_csv_list.txt

TEST_TS_CSV_LIST=$CONFIGS_DIR/test_ts_csv_list.txt

TRAIN_DATA_DIR=$CLOUD_ROOT_DIR/data/video_data/single_video_annotations/

PRETRAINED_NN_PATH=$CLOUD_ROOT_DIR/scratch_results/keras_offloader_DNN/offloader_DNN_model/

# train the offloader DNN
python3 test_streaming_offload_NN.py --output-results-dir $OUTPUT_RESULTS_DIR  --predict-var $PREDICT_VAR --train-features-list $TRAIN_FEATURES_LIST --train-ts-csv-list $TRAIN_TS_CSV_LIST --test-ts-csv-list $TEST_TS_CSV_LIST --train-data-dir $TRAIN_DATA_DIR --pretrained-NN-path $PRETRAINED_NN_PATH

