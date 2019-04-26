import pandas
import numpy
import sys,os
from collections import OrderedDict
import argparse

base_video_dir = os.environ['CLOUD_ROOT_DIR']
utils_dir = base_video_dir + '/utils/'
sys.path.append(utils_dir)

from DNN_offloader_utils import *
from textfile_utils import *
from plotting_utils import * 

if __name__ == "__main__":

    # arguments
    parser = argparse.ArgumentParser(description='DNNOffloader')
    parser.add_argument('--output-results-dir', type=str, required=False)
    parser.add_argument('--predict-var', type=str, required=False, default = 'offload')
    parser.add_argument('--train-features-list', type=str, required=False)
    parser.add_argument('--train-ts-csv-list', type=str, required=False)
    parser.add_argument('--test-ts-csv-list', type=str, required=False)
    parser.add_argument('--train-data-dir', type=str, required=False)
    parser.add_argument('--pretrained-NN-path', type=str, required=False)

    args = parser.parse_args()

    # train ts
    train_ts_csv_list = list_from_file(args.train_ts_csv_list)

    # test ts
    test_ts_csv_list = list_from_file(args.test_ts_csv_list)

    # frame_number  SVM_confidence  prediction      face_confidence embedding_distance      frame_diff_val

    # training data
    train_data_dir = args.train_data_dir

    # where results go
    prediction_data_dir = args.output_results_dir + '/test_predictions_offloader_DNN/'
    remove_and_create_dir(prediction_data_dir)

    total_csv_list = train_ts_csv_list + test_ts_csv_list
    # create training dataframe for keras

    # features to train on
    train_features_list = args.train_features_list

    train_features = list_from_file(train_features_list)

    # output var to predict
    var_to_predict = [args.predict_var]

    # where neural net results should go
    NN_PATH = args.pretrained_NN_path

    # load the model
    NN_model = load_keras_model(save_dir = NN_PATH, prefix = 'offload')

    feature_scaler = load_scaler(out_dir = NN_PATH)

    PRINT_MODE = False

    for i, ts_csv in enumerate(total_csv_list):
        print(' ')
        print('RUNNING OFFLOADER NETWORK')
        print('csv_name: ', ts_csv)

        raw_ts_csv = train_data_dir + '/train_' + ts_csv
        
        predict_ts_csv = prediction_data_dir + '/withPredict_' + ts_csv
        
        out_file = open(predict_ts_csv, 'w')

        correct_vec = []

        with open(raw_ts_csv, 'r') as f:
            for i, line in enumerate(f):
                    # just get the header
                    if i == 0:
                        header = line.strip().split()

                        # new columns: 
                        new_header = header + ['predict_offload_decision', 'binary_offload_decision', 'offload_correct']
                        out_file.write('\t'.join(new_header) + '\n')

                    if i > 0:
                        split_line = line.strip().split()

                        row_data = OrderedDict(zip(header, split_line))
                        
                        row_data_pandas = {}
                        for k,v in row_data.items():
                            row_data_pandas[k] = [v]

                        row_df = pandas.DataFrame(row_data_pandas)

                        # scale the prediction
                        train_features_matrix = row_df[train_features].as_matrix()

                        scaled_input_features = feature_scaler.transform(train_features_matrix)

                        # predicted decision
                        predict_offload_decision = NN_model.predict(scaled_input_features)[0][0]

                        binary_offload_decision = int(predict_offload_decision > 0.50)

                        # compare accuracy 	
                        true_offload_decision = int(row_data['offload'])
                     
                        offload_correct = (true_offload_decision == binary_offload_decision)
                            
                        correct_vec.append(offload_correct)

                        # frame_number	SVM_confidence	prediction	face_confidence	embedding_distance	frame_diff_val	correct	unknown	numeric_prediction	offload	num_detect

                        # new columns: predict_offload_decision, binary_offload_decision, offload_correct
                        out_line = split_line + [str(predict_offload_decision), str(binary_offload_decision), str(offload_correct)]
                        out_file.write('\t'.join(out_line) + '\n')

                        # now write to the output file
                        if PRINT_MODE:
                            print(' ')
                            print('frame: ', row_data['frame_number'], 'offload_correct: ', offload_correct)
                            print('true: ', true_offload_decision, 'predict: ', predict_offload_decision)
                            print('pred: ', row_data['prediction'], 'conf: ', row_data['SVM_confidence'], 'correct: ', row_data['correct'], 'unknown: ', row_data['unknown'], 'num_detect', row_data['num_detect'])
                            print(' ')
        
        percentage_accuracy = np.mean(correct_vec)
        print('OVERALL ACCURACY: ', percentage_accuracy) 
        print(' ')
