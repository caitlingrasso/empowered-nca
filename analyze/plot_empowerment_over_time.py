from cProfile import label
import numpy as np
from glob import glob
import pickle
import matplotlib.pyplot as plt
import sys,os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from genome import Genome
from visualizations import save_movie
import constants

FOLDER_NAME = 'data' # name of folder in empowered-nca/ containing results
GENS = 2000

DIR = 'data/'+FOLDER_NAME

txs = ['error', 'error_phase1_error_phase2', 'error_MI', 'MI']
labels = ['bi-loss','tri-loss','tri-loss-empowerment','bi-empowerment']

line_handles=[]
for j,tx in enumerate(txs):

    results_dir = '{}/{}/*.p'.format(DIR, tx)
    filenames = glob(results_dir)

    all_MI = np.zeros(shape=(len(filenames), GENS+1))

    for i,fn in enumerate(filenames):

        with open(fn, 'rb') as f:
            # print(fn)
            best, stats = pickle.load(f)

            if tx=='error'or tx=='MI':
                fits_per_gen, pareto_front, pf_sizes_per_gen, best_inds_per_gen = stats # unpack stats
            else:
                fits_per_gen, obj1_score_per_gen, obj2_score_per_gen, pareto_front, pf_sizes_per_gen, best_inds_per_gen = stats # unpack stats
        
        # Get the empowerment of the best 'fit' (lowest error) individual in each generation
        obj2_of_best_fit = np.zeros(len(best_inds_per_gen))
        for k,ind in enumerate(best_inds_per_gen):
            obj2_of_best_fit[k] = ind.get_objective('MI')*-1 # make a maximizing function

        all_MI[i,:]=obj2_of_best_fit

    avg_MI = np.mean(all_MI, axis=0)

    # 95% confidence intervals
    ci = 1.96 * (np.std(all_MI, axis=0)/np.sqrt(len(all_MI)))

    x = np.arange(len(avg_MI))
    line, = plt.plot(avg_MI, color=constants.color_treatment_dict[tx], linewidth=4)
    plt.fill_between(x, (avg_MI-ci), (avg_MI+ci), color=constants.color_treatment_dict[tx], alpha=.25)

    line_handles.append(line)

plt.legend(line_handles, labels, fontsize=15)
plt.xlabel('Generations', fontsize=25)
plt.ylabel('Empowerment', fontsize=25)
# plt.show()

os.makedirs('results/{}'.format(FOLDER_NAME), exist_ok=True)

plt.savefig('results/{}/avg_empowerment_curves_CI.png'.format(FOLDER_NAME), dpi=300, bbox_inches='tight')
