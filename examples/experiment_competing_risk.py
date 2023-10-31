
# Comparsion models for competing risks
# In this script we train the different models for competing risks
import sys
from nfg import datasets
from experiment import *

random_seed = 0

# Open dataset
dataset = sys.argv[1] # FRAMINGHAM, SYNTHETIC_COMPETING, PBC, SEER

# Specific fold selection
fold = None
if len(sys.argv) == 3:
    fold = int(sys.argv[2])

print("Script running experiments on ", dataset)
x, t, e, covariates = datasets.load_dataset(dataset, competing = True) 

# Hyperparameters
max_epochs = 1000
grid_search = 100
layers = [[i] * (j + 1) for i in [25, 50] for j in range(4)]
layers_large = [[i] * (j + 1) for i in [25, 50] for j in range(8)]

batch = [100, 250] if dataset != 'SEER' else [1000, 5000]

# DSM
param_grid = {
    'epochs': [max_epochs],
    'learning_rate' : [1e-3, 1e-4],
    'batch': batch,

    'k' : [2, 3, 4, 5],
    'distribution' : ['LogNormal', 'Weibull'],
    'layers' : layers_large,
}
DSMExperiment.create(param_grid, n_iter = grid_search, path = 'Results/{}_dsm'.format(dataset), random_seed = random_seed, fold = fold).train(x, t, e)

# NFG Competing risk
param_grid = {
    'epochs': [max_epochs],
    'learning_rate' : [1e-3, 1e-4],
    'batch': batch,
    
    'dropout': [0., 0.25, 0.5, 0.75],

    'layers_surv': layers,
    'layers' : layers,
    'act': ['Tanh'],
}
NFGExperiment.create(param_grid, n_iter = grid_search, path = 'Results/{}_nfg'.format(dataset), random_seed = random_seed, fold = fold).train(x, t, e)
NFGExperiment.create(param_grid, n_iter = grid_search, path = 'Results/{}_nfgcs'.format(dataset), random_seed = random_seed, fold = fold).train(x, t, e, cause_specific = True)

# Compared for fixed uncertainty
param_grid['dropout'] = [0.25]
NFGExperiment.create(param_grid, n_iter = grid_search, path = 'Results/{}_nfgfu'.format(dataset), random_seed = random_seed, fold = fold).train(x, t, e)
NFGExperiment.create(param_grid, n_iter = grid_search, path = 'Results/{}_nfgcsfu'.format(dataset), random_seed = random_seed, fold = fold).train(x, t, e, cause_specific = True)


# Desurv
param_grid = {
    'epochs': [max_epochs],
    'learning_rate' : [1e-3, 1e-4],
    'batch': batch,

    'layers_surv': layers_large,
    'layers': layers_large,
    'act': ['Tanh'],
}
DeSurvExperiment.create(param_grid, n_iter = grid_search, path = 'Results/{}_ds'.format(dataset), random_seed = random_seed).train(x, t, e)

# DeepHit Competing risk
param_grid = {
    'epochs': [max_epochs],
    'learning_rate' : [1e-3, 1e-4],
    'batch': batch,

    'nodes' : layers,
    'shared' : layers
}
DeepHitExperiment.create(param_grid, n_iter = grid_search, path = 'Results/{}_dh'.format(dataset), random_seed = random_seed, fold = fold).train(x, t, e)