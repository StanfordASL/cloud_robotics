import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import random
import itertools
import sys,os
import copy

#np.random.seed(42)
import numpy as np
from scipy.ndimage.interpolation import shift





"""
    shift a ts by one to the right
"""

def shift_dict_value(input_dict = None, key = None, shift_amount = -1, fill_value = np.NaN):

    input_vec = input_dict[key]

    shifted_vec = shift_np_array(input_vec = input_vec, shift_amount = shift_amount, fill_value = fill_value)

    input_dict[key] = list(shifted_vec)

    return input_dict




"""
    shift an array by one to the left
"""

def shift_np_array(input_vec = None, shift_amount = -1, fill_value = np.NaN):

    new_vec = copy.deepcopy(input_vec) 

    return shift(new_vec, shift_amount, cval=fill_value)

def print_dict(input_dict):
    for k,v in input_dict.items():
        print(k, v)

"""
    merge a list of lists
"""

def flatten_list(input_list):

    flat_list = list(itertools.chain.from_iterable(input_list))
    return flat_list



"""
sort a dictionary by value
"""

def sort_dict_value(sample_dict = None, reverse_mode = False, sort_index = 1):

    sorted_dict = sorted(sample_dict.items(), key=lambda x: x[sort_index], reverse = reverse_mode)

    return sorted_dict

"""
write a dict to a pkl file for saving
"""

def write_pkl(fname = None, input_dict = None):
    with open(fname, 'wb') as f:
        pickle.dump(input_dict, f)

"""
load the contents of a pkl file

"""

def load_pkl(fname = None):
    with open(fname, 'rb') as f:
        out_dict = pickle.load(f)
    return out_dict

def reverse_keys_values_dict(input_dict = None):

    output_dict = {}

    for k,v in input_dict.items():
        output_dict[v] = k

    return output_dict

def remove_and_create_dir(path):
    """ System call to rm -rf and then re-create a dir """

    dir = os.path.dirname(path)
    print('attempting to delete ', dir, ' path ', path)
    if os.path.exists(path):
        os.system("rm -rf " + path)
    os.system("mkdir -p " + path)





