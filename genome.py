import random
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pyinform

from ca_model import CA_MODEL
import constants

class Genome:

    def __init__(self, id, weights=None, init_grid=None, init_signal=None):
        self.genome = CA_MODEL(weights=weights, init_grid=init_grid, init_signal=init_signal)
        self.id = id

        # objective scores
        self.age = 0
        self.error = 0
        self.error_phase1 = 0 # split into before and after damage
        self.error_phase2 = 0
        self.MI = 0

    def evaluate(self, objectives, target):

        history, sensors, actions = self.genome.run()

        # Compute Objectives

        # Always compute error and empowerment
        self.error = self.evaluate_error(history, target) # max fitness
        
        # if 'MI' in objectives:
        # split into Ank and Sn2k
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

        self.MI = self.compute_MI(X=Ank, Y=Sn2k, local=False)


        if 'error_phase1' in objectives and 'error_phase2' in objectives:
            split = constants.ITERATIONS//2
            history1 = history[:split]
            history2 = history[split:]
            self.error_phase1 = self.evaluate_error(history1, target)
            self.error_phase2 = self.evaluate_error(history2, target)

    def evaluate_error(self, history, target):
        '''
        Computes error between current grid and target grid averaged over all iterations
        Normalized between 0-1
        Error is to be minimized
        '''
        all_fits = np.zeros(len(history))
        for i in range(len(history)):
            all_fits[i] = self.evaluate_grid_diff(history[i][:,:,0],target)/(constants.GRID_SIZE**2) # normalized grid difference
        return np.mean(all_fits)

    def evaluate_grid_diff(self, x, target):
        # computes L2 loss on single grid (difference measure)
        return np.sum(np.power((x-target),2))

    def compute_MI(self, X,Y,local=False):
        MI = pyinform.mutual_info(xs=X, ys=Y,local=local)
        return MI*-1

    def mutate(self):
        self.genome.mutate()

    def dominates_other(self, other, objectives):
        # Note: age is always passed in as the first objective

        if len(objectives) == 2:  # bi-objective

            obj1 = objectives[0]
            obj2 = objectives[1]

            if self.get_objective(obj1) == other.get_objective(obj1) and self.get_objective(
                    obj2) == other.get_objective(obj2):
                return self.id > other.id
            elif self.get_objective(obj1) <= other.get_objective(obj1) and self.get_objective(
                    obj2) <= other.get_objective(obj2):
                return True
            else:
                return False

        elif len(objectives) == 3:  # tri-objective

            obj1 = objectives[0]
            obj2 = objectives[1]
            obj3 = objectives[2]

            if self.get_objective(obj1) == other.get_objective(obj1) and self.get_objective(
                    obj2) == other.get_objective(obj2) and self.get_objective(obj3) == other.get_objective(obj3):
                return self.id > other.id
            elif self.get_objective(obj1) <= other.get_objective(obj1) and self.get_objective(
                    obj2) <= other.get_objective(obj2) and self.get_objective(obj3) <= other.get_objective(obj3):
                return True
            else:
                return False

    def get_objective(self, objective):
        if objective == 'age':
            return self.age
        elif objective == 'error':
            return self.error
        elif objective == 'error_phase1':
            return self.error_phase1
        elif objective == 'error_phase2':
            return self.error_phase2
        elif objective == 'MI':
            return self.MI

    def print(self, objectives):
        print('[id:', self.id, end=' ')
        for objective in objectives:
            print(objective, ':', self.get_objective(objective), end=' ')
        if 'error' not in objectives:
            print('error:', self.error, end=' ')
        print(']', end='')

    def playback(self, iterations=constants.ITERATIONS, return_SA = False):

        history, sensors_timeseries, actions_timeseries = self.genome.run(iterations=iterations)
        if return_SA:
            return history, sensors_timeseries, actions_timeseries
        else:
            return history

