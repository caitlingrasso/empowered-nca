import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import argparse
import time
import os
import pickle

from numpy.lib.function_base import disp
from numpy.lib.npyio import save

from config import targets, init
import constants
from optimizer import Optimizer
from visualizations import display_body_signal, display_grid

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=1, type=int)
    parser.add_argument('--gens', default=constants.GENERATIONS, type=int)
    parser.add_argument('--popsize', default=constants.POP_SIZE, type=int)
    parser.add_argument('--target', default='square')
    parser.add_argument('--objective1', default='error')
    parser.add_argument('--objective2', default=None)
    parser.add_argument('--checkpoint_every', default=500, type=int)
    return parser.parse_args(args)

if __name__=="__main__":

    start_time = time.time()

    args = sys.argv[1:]
    args = parse_args(args)

    # set seed
    np.random.seed(args.run)
    random.seed(args.run)

    # set objectives
    if args.objective2 is None or args.objective2=='None':
        objectives= ['age', args.objective1] 
        obj2 = ''
    else:
        objectives = ['age', args.objective1, args.objective2]
        obj2 = args.objective2+'_'
    
    # # Save info of every individual for AFPO rainbow waterfall plot for first run only
    # if args.run==1: 
    #     constants.SAVE_ALL=True

    # set target
    target_str = args.target
    targ=targets[target_str]

    # Plotting initial and target grid conditions
    # signal = np.zeros(targ.shape)
    # targ = np.concatenate((targ[:,:,None], signal[:,:,None]), axis=2)
    # signal[constants.GRID_SIZE//2,constants.GRID_SIZE//2]=255
    # init = np.concatenate((init[:,:,None], signal[:,:,None]), axis=2)
    # display_body_signal(init, save=True, fn='shapes/seed_{}_initial_condition.png'.format(constants.GRID_SIZE))
    # display_body_signal(targ, save=True, fn='shapes/{}_{}_target.png'.format(target_str, constants.GRID_SIZE))
    # exit()

    # Filenames

    if args.objective2=='None':
        save_path = 'data/{}'.format(args.objective1)
    else:
        save_path = 'data/{}_{}'.format(args.objective1, args.objective2)
    os.makedirs(save_path, exist_ok=True)
    
    prefix = '{}_{}{}{}_{}gens_{}ps_{}i_run{}'.format(args.objective1, obj2, target_str,constants.GRID_SIZE, args.gens, args.popsize, \
        constants.ITERATIONS, args.run)
    save_filename = '{}/{}.p'.format(save_path, prefix)
    save_all_dir = 'data/{}_all_inds/'.format(prefix)
    checkpoint_dir = 'data/{}_checkpoints/'.format(prefix)

    # make directory for checkpoints
    if checkpoint_dir is not None and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # START RUN
    optimizer = Optimizer(target=targ, objectives=objectives, gens=args.gens, pop_size=args.popsize, 
                        checkpoint_every=args.checkpoint_every, save_all_dir=save_all_dir,
                        checkpoint_dir=checkpoint_dir, run_nbr=args.run, init_grid=init)

    best, stats = optimizer.run() 
    
    # For bi-objective:
    # stats = [fits_per_gen, pareto_front, pf_sizes_per_gen, best_inds_per_gen]
    
    # For tri-objective:
    # stats = [fits_per_gen, obj1_score_per_gen, obj2_score_per_gen, pareto_front, pf_sizes_per_gen, best_inds_per_gen]
    
    # Save data from run 
    f = open(save_filename, 'wb')
    pickle.dump([best,stats], f)
    f.close()

    end_time = time.time()
    print('--', (end_time-start_time)/3600, 'hours --')  
