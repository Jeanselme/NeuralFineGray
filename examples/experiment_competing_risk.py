
# Comparsion models for competing risks
# In this script we train the different models for competing risks
import sys
from nfg import datasets
from experiment import *

random_seed = 0

# Open dataset
dataset = sys.argv[1] # FRAMINGHAM, SYNTHETIC_COMPETING, PBC
print("Script running experiments on ", dataset)
x, t, e, covariates = datasets.load_dataset(dataset, competing = True) 

# Hyperparameters and evaluations
horizons = [0.25, 0.5, 0.75]
times = np.quantile(t[e!=0], horizons)

max_epochs = 1000
grid_search = 100
layers = [[i] * (j + 1) for i in [50, 100] for j in range(3)]
layers_large = [[i] * (j + 1) for i in [50, 100] for j in range(6)]

# Models
## Save data for R 
kf = StratifiedKFold(random_state = random_seed, shuffle = True)
data = pd.DataFrame(x, columns = covariates)
for i, (train_index, test_index) in enumerate(kf.split(x, e)):
    train_index, dev_index = train_test_split(train_index, test_size = 0.2, random_state = random_seed, stratify = e[train_index])
    dev_index, val_index   = train_test_split(dev_index,   test_size = 0.5, random_state = random_seed, stratify = e[dev_index])

    # Keep track of the whole indexing
    fold = pd.Series(0, index = data.index)
    fold[train_index] = "Train"
    fold[dev_index] = "Dev"
    fold[val_index] = "Val"
    fold[test_index] = "Test"
    data['Fold_{}'.format(i)] = fold

data['Time'] = t
data['Event'] = e
data.to_csv('data/' + dataset + '.csv', index = False)

# DSM
param_grid = {
    'epochs': [max_epochs],
    'learning_rate' : [1e-3, 1e-4],
    'batch': [100, 250],

    'k' : [2, 3, 4, 5],
    'distribution' : ['LogNormal', 'Weibull'],
    'layers' : layers_large,
}
DSMExperiment.create(param_grid, n_iter = grid_search, path = 'Results/{}_dsm'.format(dataset), times = times, random_seed = random_seed).train(x, t, e)
DSMExperiment.create(param_grid, n_iter = grid_search, path = 'Results/{}_dsmcs'.format(dataset), times = times, random_seed = random_seed).train(x, t, e, cause_specific = True)

# NFG Competing risk
param_grid = {
    'epochs': [max_epochs],
    'learning_rate' : [1e-3, 1e-4],
    'batch': [100, 250],

    'layers_surv': layers,
    'layers' : layers,
    'act': ['Tanh'],
}
NFGExperiment.create(param_grid, n_iter = grid_search, path = 'Results/{}_nfg'.format(dataset), times = times, random_seed = random_seed).train(x, t, e)
NFGExperiment.create(param_grid, n_iter = grid_search, path = 'Results/{}_nfgcs'.format(dataset), times = times, random_seed = random_seed).train(x, t, e, cause_specific = True)

# DeepHit Competing risk
param_grid = {
    'epochs': [max_epochs],
    'learning_rate' : [1e-3, 1e-4],
    'batch': [100, 250],

    'nodes' : layers,
    'shared' : layers
}
DeepHitExperiment.create(param_grid, n_iter = grid_search, path = 'Results/{}_dh'.format(dataset), times = times, random_seed = random_seed).train(x, t, e)
DeepHitExperiment.create(param_grid, n_iter = grid_search, path = 'Results/{}_dhcs'.format(dataset), times = times, random_seed = random_seed).train(x, t, e, cause_specific = True)
