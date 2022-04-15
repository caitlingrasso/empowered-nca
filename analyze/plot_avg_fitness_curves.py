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

    all_fits = np.zeros(shape=(len(filenames), GENS+1))

    for i,fn in enumerate(filenames):

        with open(fn, 'rb') as f:
            best, stats = pickle.load(f)

            if tx=='error'or tx=='MI':
                fits_per_gen, pareto_front, pf_sizes_per_gen, best_inds_per_gen = stats # unpack stats
            else:
                fits_per_gen, obj1_score_per_gen, obj2_score_per_gen, pareto_front, pf_sizes_per_gen, best_inds_per_gen = stats # unpack stats

        all_fits[i,:]=fits_per_gen

    avg_fits = np.mean(all_fits, axis=0)
    # 95% confidence intervals
    ci = 1.96 * (np.std(all_fits, axis=0)/np.sqrt(len(all_fits)))

    x = np.arange(len(avg_fits))
    # line, = plt.plot(avg_fits, color=constants.color_treatment_dict[tx], linewidth=4)
    line, = plt.semilogy(avg_fits, color=constants.color_treatment_dict[tx], linewidth=4)
    plt.fill_between(x, (avg_fits-ci), (avg_fits+ci), color=constants.color_treatment_dict[tx], alpha=.25)

    line_handles.append(line)

plt.legend(line_handles, labels, fontsize=15, loc='center right')
plt.xlabel('Generations', fontsize=25)
plt.ylabel('Loss', fontsize=25)
# plt.show()

os.makedirs('results/', exist_ok=True)
os.makedirs('results/{}'.format(FOLDER_NAME), exist_ok=True)

plt.savefig('results/{}/avg_fitness_curves_CI_semilog.png'.format(FOLDER_NAME), dpi=300, bbox_inches='tight')