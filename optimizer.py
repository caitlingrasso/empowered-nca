import os
import numpy as np
import copy
import operator
import random
import pickle

import constants
from genome import Genome

class Optimizer:

    def __init__(self, target, objectives=['age', 'error'], pop_size=constants.POP_SIZE, gens=constants.GENERATIONS,
                 save_all_dir=None, checkpoint=True, checkpoint_dir='', checkpoint_every=1, run_nbr=1, init_grid=None, init_signal=None):

        # evolution parameters
        self.gens = gens
        self.target_size = pop_size
        self.next_id = 0
        self.gen=0
        self.checkpoint = checkpoint
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = checkpoint_every
        self.run_nbr=run_nbr

        # cellular automata parameters
        self.target = target
        self.init_grid = init_grid
        self.init_signal = init_signal

        # objectives (age is always first objective)
        self.objectives = objectives
    
        # saving data
        self.fits_per_gen = np.zeros(self.gens+1)
        self.pf_sizes_per_gen = np.zeros(self.gens+1)
        self.best_inds_per_gen = []
        self.save_dir = save_all_dir

        if len(objectives)==3:
            self.triobjective = True
            self.obj1_score_per_gen = np.zeros(self.gens+1)
            self.obj2_score_per_gen = np.zeros(self.gens+1)
        else:
            self.triobjective = False
        
        self.population = self.create_initial_population(pop_size)

    def run(self, continue_from_checkpoint=False, additional_gens=1):
        if not continue_from_checkpoint:
            self.perform_first_generation()

            while self.gen < self.gens + 1:
                self.perform_one_generation()
                if self.gen % self.checkpoint_every == 0 and self.checkpoint:
                    self.save_checkpoint()
                self.gen += 1
        else:
            max_gens = self.gen + additional_gens

            new_fits = np.zeros(max_gens + 1)
            new_fits[0:len(self.fits_per_gen)] = self.fits_per_gen
            self.fits_per_gen = new_fits

            new_pf_sizes = np.zeros(max_gens + 1)
            new_pf_sizes[0:len(self.pf_sizes_per_gen)] = self.pf_sizes_per_gen
            self.pf_sizes_per_gen = new_pf_sizes

            if self.triobjective:
                new_obj1_score_per_gen = np.zeros(max_gens + 1)
                new_obj1_score_per_gen[0:len(self.obj1_score_per_gen)] = self.obj1_score_per_gen
                self.obj1_score_per_gen = new_obj1_score_per_gen

                new_obj2_score_per_gen = np.zeros(max_gens + 1)
                new_obj2_score_per_gen[0:len(self.obj2_score_per_gen)] = self.obj2_score_per_gen
                self.obj2_score_per_gen = new_obj2_score_per_gen

            self.gen += 1

            while self.gen < max_gens + 1:
                self.perform_one_generation()
                if self.gen % self.checkpoint_every == 0 and self.checkpoint:
                    self.save_checkpoint()
                self.gen += 1

        best = self.find_best()
        pareto_front = self.find_pareto_front_individuals()

        if self.triobjective:
            stats = [self.fits_per_gen, self.obj1_score_per_gen, self.obj2_score_per_gen, pareto_front, self.pf_sizes_per_gen, self.best_inds_per_gen]
        else:
            stats = [self.fits_per_gen, pareto_front, self.pf_sizes_per_gen, self.best_inds_per_gen]

        return best, stats

    def create_initial_population(self, pop_size):
        population = {}
        for i in range(pop_size):
            population[i] = Genome(self.next_id, init_grid=self.init_grid, init_signal=self.init_signal)
            self.next_id += 1
        return population

    def perform_first_generation(self):
        self.population = self.evaluate(self.population)
        self.print_best()

        # record stats
        self.pf_sizes_per_gen[0] = len(self.find_pareto_front())
        best = self.find_best()
        self.fits_per_gen[0] = best.get_objective('error')
        self.best_inds_per_gen.append(best)
        if self.triobjective:
            self.obj1_score_per_gen[0] = self.find_best(objective=self.objectives[1]).get_objective(self.objectives[1])
            self.obj2_score_per_gen[0] = self.find_best(objective=self.objectives[2]).get_objective(self.objectives[2])

        if constants.SAVE_ALL:
            self.save_all(0)
        self.gen += 1

    def perform_one_generation(self):
        self.increase_age()
        children = self.breed()
        children = self.insert_random(children)
        children = self.evaluate(children)
        self.extend(children)
        pf = self.reduce_pop()
        self.print_best()
        
        # record stats
        self.pf_sizes_per_gen[self.gen] = len(pf)
        best = self.find_best()
        self.fits_per_gen[self.gen] = best.get_objective(objective='error')
        self.best_inds_per_gen.append(best)
        if self.triobjective:
            self.obj1_score_per_gen[self.gen] = self.find_best(objective=self.objectives[1]).get_objective(self.objectives[1])
            self.obj2_score_per_gen[self.gen] = self.find_best(objective=self.objectives[2]).get_objective(self.objectives[2])

        if constants.SAVE_ALL:
            self.save_all(self.gen)

    def evaluate(self, population):
        for i in range(len(population)):
            population[i].evaluate(self.objectives, target=self.target)
        return population

    def print_best(self):
        print(self.gen, end=' ')
        best = self.find_best()
        best.print(self.objectives)
        print()
        
    def print_population(self):    
        print('------------------------------------------------------------------') 
        print(self.gen, end=' ')
        for p in self.population:
            self.population[p].print(self.objectives)
            print()

    def breed(self):
        pop_size = len(self.population)
        children = []
        for i in range(pop_size):
            parent = self.tournament_selection()
            child = copy.deepcopy(self.population[parent])
            child.id = self.next_id
            self.next_id += 1
            child.mutate()
            children.append(child)
        return children

    def insert_random(self, children):
        children.append(Genome(self.next_id, init_grid=self.init_grid, init_signal=self.init_signal))
        self.next_id += 1
        return children

    def extend(self, children):
        pop_size = len(self.population)
        for i, j in enumerate(range(pop_size, pop_size + len(children))):
            self.population[j] = children[i]

    def tournament_selection(self):
        p1 = np.random.randint(self.target_size)
        p2 = np.random.randint(self.target_size)
        while p1 == p2:
            p2 = np.random.randint(self.target_size)

        if self.population[p1].error < self.population[p2].error:
            return p1
        else:
            return p2

    def increase_age(self):
        for i in self.population:
            self.population[i].age += 1

    def reduce_pop(self):

        pf = self.find_pareto_front()
        pareto_size = len(pf)

        if pareto_size > self.target_size:
            self.target_size = pareto_size

        # remove dominated individuals until the target population size is reached
        while len(self.population) > self.target_size:

            pop_size = len(self.population)

            ind1 = np.random.randint(pop_size)
            ind2 = np.random.randint(pop_size)
            while ind1 == ind2:
                ind2 = np.random.randint(pop_size)

            if self.dominates(ind1, ind2):  # ind1 dominates

                for i in range(ind2, len(self.population) - 1):
                    self.population[i] = self.population.pop(i + 1)

            elif self.dominates(ind2, ind1):  # ind2 dominates

                for i in range(ind1, len(self.population) - 1):
                    self.population[i] = self.population.pop(i + 1)
        return pf

    def dominates(self, ind1, ind2):
        return self.population[ind1].dominates_other(self.population[ind2], self.objectives)

    def find_best(self, objective='error'):
        # Finds best individual based on error
        sorted_pop = sorted(self.population.values(), key=operator.attrgetter(objective), reverse=False)
        return sorted_pop[0]

    def find_pareto_front(self):
        #  Returns indices of non-dominated individuals in the population

        pareto_front = []

        for i in self.population:
            i_is_dominated = False
            for j in self.population:
                if i != j:
                    if self.dominates(j, i):
                        i_is_dominated = True
            if not i_is_dominated:
                pareto_front.append(i)

        return pareto_front

    def find_pareto_front_individuals(self):
        pf = self.find_pareto_front()
        pf_inds = []
        for i in pf:
            pf_inds.append(self.population[i])
        return pf_inds

    def save_all(self, g):
        if self.save_dir is not None and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # saves the population to a text file named Gen_gen.txt
        if g < 10:  # single digit
            filename = "{}Gen_000{}.txt".format(self.save_dir, g)
        elif g < 100:  # double digit
            filename = "{}Gen_00{}.txt".format(self.save_dir, g)
        elif g < 1000:  # double digit
            filename = "{}Gen_0{}.txt".format(self.save_dir, g)
        else:
            filename = "{}Gen_{}.txt".format(self.save_dir, g)

        f = open(filename, "w+")
        if len(self.objectives) == 2:
            f.write("{}\t{}\n".format(self.objectives[0], self.objectives[1]))
            for i in self.population:
                line = "{}\t{}\n".format(self.population[i].get_objective(self.objectives[0]),
                                         self.population[i].get_objective(self.objectives[1]))
                f.write(line)
        elif len(self.objectives) == 3 and "error" in self.objectives:
            f.write("{}\t{}\t{}\n".format(self.objectives[0], self.objectives[1], self.objectives[2]))
            for i in self.population:
                line = "{}\t{}\t{}\n".format(self.population[i].get_objective(self.objectives[0]),
                                            self.population[i].get_objective(self.objectives[1]),
                                            self.population[i].get_objective(self.objectives[2]))
                f.write(line)
        else:
            f.write("{}\t{}\t{}\t{}\n".format(self.objectives[0], "error", self.objectives[1], self.objectives[2]))
            for i in self.population:
                line = "{}\t{}\t{}\t{}\n".format(self.population[i].get_objective(self.objectives[0]),
                                            self.population[i].error,
                                            self.population[i].get_objective(self.objectives[1]),
                                            self.population[i].get_objective(self.objectives[2]))
                f.write(line)

        f.close()

    def save_checkpoint(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        filename = '{}run{}_{}gens.p'.format(self.checkpoint_dir, self.run_nbr, self.gen)
        print('SAVING POPULATION IN: ', filename)
        rng_state = random.getstate()
        np_rng_state = np.random.get_state()
        f = open(filename, 'wb')
        pickle.dump([self, rng_state, np_rng_state], f)
        f.close()
