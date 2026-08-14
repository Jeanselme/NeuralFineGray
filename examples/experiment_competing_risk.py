
# Comparsion models for competing risks
# In this script we train the different models for competing risks
import sys
sys.path.append('NeuralFineGray/')
sys.path.append('NeuralFineGray/DeepSurvivalMachines/')

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
times = np.sort(np.linspace(t.min(), t.max(), 100).tolist() + [365, 365*5, 3650])

# NFG Competing risk
param_grid = {
    'epochs': [max_epochs],
    'learning_rate' : [1e-3, 1e-4],
    'batch': batch,
    
    'dropout': [0., 0.25, 0.5, 0.75],

    'layers_surv': layers,
    'layers' : layers,
    'act': ['Tanh'],

    'normalise': ['minmax']
}
NFGExperiment.create(param_grid, times = times, n_iter = grid_search, path = 'Results/{}_nfg'.format(dataset), random_seed = random_seed, fold = fold).train(x, t, e)
NFGExperiment.create(param_grid, times = times, n_iter = grid_search, path = 'Results/{}_nfgnc'.format(dataset), random_seed = random_seed, fold = fold).train(x, t, e == 1)

# Desurv
param_grid = {
    'epochs': [max_epochs],
    'learning_rate' : [1e-3, 1e-4],
    'batch': batch,

    'embedding': [True],
    'layers_surv': layers,
    'layers': layers,
    'act': ['Tanh'],

    'normalise': ['minmax']
}
DeSurvExperiment.create(param_grid, n_iter = grid_search, path = 'Results/{}_ds'.format(dataset), random_seed = random_seed, fold = fold).train(x, t, e)
DeSurvExperiment.create(param_grid, n_iter = grid_search, path = 'Results/{}_dsnc'.format(dataset), random_seed = random_seed, fold = fold).train(x, t, e == 1)

# DeepHit Competing risk
param_grid = {
    'epochs': [max_epochs],
    'learning_rate' : [1e-3, 1e-4],
    'batch': batch,

    'nodes' : layers,
    'shared' : layers
}
DeepHitExperiment.create(param_grid, n_iter = grid_search, path = 'Results/{}_dh'.format(dataset), random_seed = random_seed, fold = fold).train(x, t, e)
DeepHitExperiment.create(param_grid, n_iter = grid_search, path = 'Results/{}_dhnc'.format(dataset), random_seed = random_seed, fold = fold).train(x, t, e == 1)