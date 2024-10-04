# Neural Fine Gray
This repository allows to reproduce the results in [Neural Fine Gray](https://arxiv.org/abs/2305.06703) - Please use the release: CHIL for exact reproduction.  
A neural network approach to the problem of competing risks, leveraging monotone neural networks to model the cumulative incidence functions.

## Model
The model consists in two neural networks: one models the cumulative incidence function and the other the balance to ensure that they add up to one.

![Model](./images/nfg.png)

## How to use ?
To use the model, one needs to execute:
```python
from nfg import NeuralFineGray
model = NeuralFineGray()
model.fit(x, t, e)
model.predict_risk(x, risk = 1)
```
With `x`, the covariates, `t`, the event times and `e`, the cause of end of follow up (0 is censoring). 
It is critical to normalise `t` to range in (0, 1), by dividing by the maximum time. Similarly, at evaluation, one must use the normalised time horizons (see `examples/` for a detailed application)

A full example with analysis is provided in `examples/Neural Fine Gray on FRAMINGHAM Dataset.ipynb`.
## Reproduce paper's results
To reproduce the paper's results:

0. Clone the repository with dependencies: `git clone git@github.com:Jeanselme/NeuralFineGray.git --recursive`
1. Create a conda environment with all necessary libraries `pycox`, `lifelines`, `pysurvival`
2. Add path `export PYTHONPATH="$PWD:$PWD/DeepSurvivalMachines:$PYTHONPATH"`
3. Run `examples/experiment_competing_risk.py FRAMINGHAM` to run all models on the `FRAMINGHAM` dataset
4. Repeat with `PBC`, `SYNTHETIC_COMPETING` and `SEER` to run on each dataset
5. Analysis using `examples/Analysis.ipynb` to measure performance

Note that you will need to export the `SEER` dataset from [https://seer.cancer.gov/data/](https://seer.cancer.gov/data/). The previous scripts allow you to reproduce all the models presented in the paper except the Fine-Gray appraoch that requires: 
0. Install R and the libraries: `riskRegression`, `prodlim`, `survival`, `cmprsk` and `readr`
1. Create a folder `data/` in `examples/` to save the generated files
2. Run `examples/process_data.py FRAMINGHAM` to create a csv files with the same data split used in the Python scripts
3. Run `examples/FineGray.R` to create the predictions of a Fine-Gray model (Note that you will need to change the content of this file for running on a subset of datasets)

## Compare to a new method
Adding a new method consists in adding a child to `Experiment` in `experiment.py` with functions to compute the nll and fit the model.
Then, add the method in `examples/experiment_competing_risk.py` and follow the previous point. 
`TODOs` have been added to make the addition of a new method easier.

# Setup
## Structure
We followed the same architecture than the [DeepSurvivalMachines](https://github.com/autonlab/DeepSurvivalMachines) repository with the model in `nfg/` - only the api should be used to test the model. Examples are provided in `examples/`. 

## Clone
```
git clone git@github.com:Jeanselme/NeuralFineGray.git --recursive
```

## Requirements
The model relies on `DeepSurvivalMachines`, `pytorch`, `numpy` and `tqdm`.  
To run the set of experiments `pycox`, `lifelines`, `pysurvival` are necessary.
