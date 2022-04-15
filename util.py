'''various helper functions'''

import numpy as np
import pickle
import operator
from glob import glob
import random
import constants
import pyinform


def sort_by(DIR, objective='error', reverse=False):
    all_files = glob(DIR)
    fits_dict = {}

    for i, file in enumerate(all_files):
        f = open(file, 'rb')
        best, stats = pickle.load(f)
        f.close()

        fits_dict[file] = best.get_objective(objective)
    if reverse:
        return sorted(fits_dict.items(), key=operator.itemgetter(1), reverse=True) # low to high (minimization)    
    else:
        return sorted(fits_dict.items(), key=operator.itemgetter(1), reverse=False) # low to high (minimization)

def continue_from_checkpoint(file, additional_gens):
    f = open(file, 'rb')
    optimizer, rng_state, np_rng_state = pickle.load(f)
    f.close()

    random.setstate(rng_state)
    np.random.set_state(np_rng_state)

    best, stats= optimizer.run(continue_from_checkpoint=True, additional_gens=additional_gens)
    return best, stats

def to_function(activation):
    if activation=='sigmoid':
        return sigmoid
    elif activation=='sin':
        return sin
    elif activation=='cos':
        return cos
    elif activation=='relu':
        return relu
    elif activation=='tanh':
        return tanh
    elif activation=='abs':
        return abs

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def sin(x):
    return np.sin(x)

def cos(x):
    return np.cos(x)

def relu(x):
    return np.maximum(np.zeros(x.shape),x)

def tanh(x):
    return np.tanh(x)

def abs(x):
    return np.abs(x)

def get_local_MI(sensors,actions,averaged=True):
    split = constants.ITERATIONS//2
    Ank = []
    Sn2k = []

    # Collect action (input) states for first half of timesteps
    for action in actions[:split]:
        at = action[:,:,-1].flatten() # last value in action list (new output signal value of cell)
        for i in range(len(at)):
            try:
                Ank[i].append(at[i])
            except:
                Ank.append([])
                Ank[i].append(at[i])

    # Collect sensor (output) states for second half of timesteps
    for sensor in sensors[split:]:
        st = np.mean(sensor[:,:,5:],axis=2).flatten() # Average last 4 signal values (input signal values of neighbors + self) - these are floats because they are averages
        for i in range(len(st)):
            try:
                Sn2k[i].append(st[i])
            except:
                Sn2k.append([])
                Sn2k[i].append(st[i])
    ################################################


    local_MI = pyinform.mutual_info(xs=Ank, ys=Sn2k, local=True)

    grid_dim = int(np.sqrt(local_MI.shape[0]))
    grids_with_padding = np.reshape(local_MI, newshape=(grid_dim,grid_dim,local_MI.shape[1]))
    empowerment_ts = grids_with_padding[1:-1, 1:-1,:]

    static_mean_cell_empowerment = np.mean(empowerment_ts, axis=2)

    if averaged:
        return static_mean_cell_empowerment
    else:
        return empowerment_ts