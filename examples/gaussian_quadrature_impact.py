
# Comparsion models for competing risks
# In this script we train the different models for competing risks
import sys
from nfg import datasets
from experiment import *

random_seed = 0

# Open dataset
dataset = sys.argv[1] # FRAMINGHAM, SYNTHETIC_COMPETING, PBC, SEER
print("Script running experiments on ", dataset)
x, t, e, covariates = datasets.load_dataset(dataset, competing = True) 

print(np.linalg.matrix_rank(x), np.linalg.matrix_rank(np.concatenate([x, t.reshape((-1, 1))], axis = 1)))

max_epochs = 1000

# DeSurv
param_grid = {
    'epochs': [max_epochs],
    'learning_rate' : [1e-3],
    'batch': [250],
    'layers' : [[50, 50, 50]],
    'layers_surv' : [[50, 50, 50]],
    'act': ['ReLU'],
}
for n in [1, 15, 100, 1000]:
    param_grid['n'] = [n]
    DeSurvExperiment.create(param_grid, n_iter = 1, path = 'Results/{}_ds_n={}'.format(dataset, n), random_seed = random_seed).train(x, t, e)