# This script extracts the different fold to then use the fine gray R script
import sys
from nfg import datasets
from experiment import *

random_seed = 0

# Open dataset
dataset = sys.argv[1] # FRAMINGHAM, SYNTHETIC_COMPETING, PBC, SEER
print("Script running experiments on ", dataset)
x, t, e, covariates = datasets.load_dataset(dataset, competing = True, drop_first = True) 

## Save data for R 
kf = StratifiedKFold(random_state = random_seed, shuffle = True)
data = pd.DataFrame(x).add_prefix('feature') # Do not save names to match R

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