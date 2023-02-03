
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

# Hyperparameters and evaluations
horizons = [0.25, 0.5, 0.75]
times = np.quantile(t[e!=0], horizons)

max_epochs = 1000
grid_search = 1

# DeSurv
param_grid = {
    'epochs': [max_epochs],
    'learning_rate' : [1e-4],
    'batch': [5000],
    
    'dropout': [0.25],

    'layers_surv': [[50] * 4],
    'layers' : [[50] * 4],
    'act': ['Tanh'],
}
for n in [2, 3, 100, 1000, 10000]:
    param_grid['n'] = [n]
    DeSurvExperiment.create(param_grid, n_iter = grid_search, path = 'Results/{}_ds_n={}'.format(dataset, n), times = times, random_seed = random_seed).train(x, t, e)

NFGExperiment.create(param_grid, n_iter = grid_search, path = 'Results/{}_nfg_fixed'.format(dataset), times = times, random_seed = random_seed).train(x, t, e)
param_grid['multihead'] = [False]
NFGExperiment.create(param_grid, n_iter = grid_search, path = 'Results/{}_nfg_onehead_fixed'.format(dataset), times = times, random_seed = random_seed).train(x, t, e)
