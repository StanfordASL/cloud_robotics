import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import random
import itertools
import sys,os
import copy
import pandas

#np.random.seed(42)
import numpy as np
from scipy.ndimage.interpolation import shift

from numpy import linalg as LA

def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))

def get_L2_norm(input_vec = None):
    return LA.norm(input_vec)

def get_RMSE(input_vec = None):

    return np.sqrt(np.mean([x**2 for x in input_vec]))

def get_random_uniform(p = 0.5):                                             
    random.seed()
    sample = np.random.uniform()                                             
    
    if sample <= p:                                                          
        return True                                                          
    else:
        return False    

"""
    pad a timeseries
"""
def get_bounded_ts_history(ts = None, t = None, NUM_PAST = None, fill_val = -1):
    
    start_time = t - NUM_PAST
    
    end_time = t

    mod_start_time = max(t-NUM_PAST, 0)

    past_vec = ts[mod_start_time: end_time]

    num_to_fill = NUM_PAST - len(past_vec)

    fill_vec = [fill_val for x in range(num_to_fill)]

    padded_vec = fill_vec + past_vec

    return padded_vec



if __name__ == '__main__':

    x = np.ones(5)

    print(x)
    L2_norm = get_L2_norm(input_vec = x)
    RMSE = get_RMSE(input_vec = x)

    print('L2: ', L2_norm)
    print('RMSE: ', RMSE)

