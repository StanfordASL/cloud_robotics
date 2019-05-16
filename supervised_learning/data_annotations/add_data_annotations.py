import pandas
import numpy
import sys,os
from collections import OrderedDict

base_video_dir = os.environ['JETSON_VIDEO_ROOT_DIR']
utils_dir = base_video_dir + '/utils/'
sys.path.append(utils_dir)

from textfile_utils import *
from plotting_utils import * 


prediction_to_numeric_dict = OrderedDict()
prediction_to_numeric_dict['unknown'] = -1
prediction_to_numeric_dict['apoorva'] = 0
prediction_to_numeric_dict['james'] = 1
prediction_to_numeric_dict['sandeep'] = 2
prediction_to_numeric_dict['joe'] = 3
prediction_to_numeric_dict['karen'] = 4
prediction_to_numeric_dict['boris'] = 5
prediction_to_numeric_dict['amine'] = 6
prediction_to_numeric_dict['andrew'] = 7
prediction_to_numeric_dict['adrian'] = 8
prediction_to_numeric_dict['trisha'] = 9
prediction_to_numeric_dict['abi'] = 10

train_ts_csv_list = ['df_SVM_output_trial_6.txt', 'df_SVM_output_trial_2.txt', 'df_SVM_output_trial_4.txt', 'df_SVM_output_trial_5.txt', 'df_SVM_output_abi_sandeep.txt']

test_csv_list = ['df_SVM_output_trial_3.txt', 'df_SVM_output_trial_1.txt', 'df_SVM_output_james_apoorva_sandeep_1224.txt']
# frame_number  SVM_confidence  prediction      face_confidence embedding_distance      frame_diff_val

raw_data_dir = 'raw_df/' 
train_data_dir = 'training_data/'

total_csv_list = [train_ts_csv_list[-1]]
total_csv_list = train_ts_csv_list + test_csv_list

# create training dataframe for keras

# add in the true annotation files

# load video annotations
###############################
video_annotations_pkl = 'pkl/video_annotations.pkl'

PRINT_MODE = False

def within_range(frame_id = None, ranges = None):
   
    return_val = None
 
    for r in ranges:
        start = int(r.split('-')[0])
        end = int(r.split('-')[1])

	float_frame = float(frame_id)

	# within range
	if ((float_frame >= start) and (float_frame <= end)):
		return r

    return return_val

annotations_dict = load_pkl(video_annotations_pkl)

total_train_df = pandas.DataFrame()

for i, ts_csv in enumerate(total_csv_list):
    print('csv_name: ', ts_csv)

    raw_ts_csv = raw_data_dir + '/' + ts_csv

    video_dict = annotations_dict[ts_csv]

    frame_ranges = video_dict.keys()

    print('keys',  frame_ranges)
    print('values',  video_dict.values())

    train_data_file = train_data_dir + '/train_' + ts_csv
    out_f = open(train_data_file, 'w') 

    local_df = pandas.read_csv(raw_ts_csv, sep = '\t')
    all_frame_numbers = list(local_df['frame_number'])

    with open(raw_ts_csv, 'r') as f:
        for i, line in enumerate(f):
                if i == 0:
                    header = line.strip().split()
		    new_header = header + ['correct', 'unknown', 'numeric_prediction', 'offload', 'num_detect']
		    out_f.write('\t'.join(new_header) + '\n')

                if i > 0:
                    split_line = line.strip().split()

		    row_data = OrderedDict(zip(header, split_line))
			
		    frame_id = row_data['frame_number']
		    conf = row_data['SVM_confidence']
		    prediction = row_data['prediction']
		
		    # frame_number  SVM_confidence  prediction      face_confidence embedding_distance      frame_diff_val
                    range_key = within_range(frame_id = frame_id, ranges = frame_ranges)
		    
		    # we have an annotation here
		    if range_key:
			true_label_list = video_dict[range_key]
			

			offload = int(prediction not in true_label_list)
			correct = int(prediction in true_label_list)
			row_data['correct'] = correct
			unknown_flag = int(prediction == 'unknown')
			row_data['unknown_flag'] = unknown_flag
			row_data['numeric_prediction'] = prediction_to_numeric_dict[prediction]
			row_data['offload'] = offload
			num_detect = all_frame_numbers.count(int(frame_id))
			row_data['num_detect'] = num_detect
 
			# now write to the output file
			if PRINT_MODE:
			    print(' ')
			    print('frame_id: ', frame_id, 'range_key: ', range_key, 'count: ', num_detect)
                            print('pred: ', prediction , 'conf: ', conf, 'true: ', true_label_list, 'correct: ', correct, 'unknown: ', unknown_flag, 'num_detect', num_detect)
			    print(' ')
			
			out_f.write('\t'.join([str(x) for x in row_data.values()]) + '\n')
			out_f.flush()

                        write_row_dict = {}
                        for k1, v1 in row_data.iteritems():
                            write_row_dict[k1] = [v1]

			# append to total train df
			total_train_df = total_train_df.append(pandas.DataFrame(write_row_dict))

    out_f.close()


# write total train df to a file
total_train_csv = train_data_dir + '/total_train_df.csv'
total_train_df.to_csv(total_train_csv)
