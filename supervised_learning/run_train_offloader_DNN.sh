INPUT_TRAIN_CSV=$CLOUD_ROOT_DIR/data/video_data/total_train_df.csv

OUTPUT_RESULTS_DIR=$CLOUD_ROOT_DIR/scratch_results/keras_offloader_DNN/

PREDICT_VAR='offload'

TRAIN_FEATURES_LIST=$PWD/offloader_DNN_features/offloader_DNN_input_features.txt

python3 train_supervised_learning_offloader.py --output-results-dir $OUTPUT_RESULTS_DIR --input-train-csv $INPUT_TRAIN_CSV --predict-var $PREDICT_VAR --train-features-list $TRAIN_FEATURES_LIST
