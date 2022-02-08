from dsm.datasets import load_dataset as load_dsm
from sklearn.preprocessing import StandardScaler
from pycox import datasets
import pandas as pd

def load_dataset(dataset='SUPPORT', **kwargs):
    if dataset == 'GBSG':
        df = datasets.gbsg.read_df()
    elif dataset == 'METABRIC':
        df = datasets.metabric.read_df()
        df = df.rename(columns = {'x0': 'MKI67', 'x1': 'EGFR', 'x2': 'PGR', 'x3': 'ERBB2', 
                                  'x4': 'Hormone', 'x5': 'Radiotherapy', 'x6': 'Chemotherapy', 'x7': 'ER-positive', 
                                  'x8': 'Age at diagnosis'})
    elif dataset == 'SYNTHETIC':
        df = datasets.rr_nl_nhp.read_df()
        df = df.drop([c for c in df.columns if 'true' in c], axis = 'columns')
    elif dataset == 'SYNTHETIC_COMPETING':
        df = pd.read_csv('https://raw.githubusercontent.com/chl8856/DeepHit/master/sample%20data/SYNTHETIC/synthetic_comprisk.csv')
        df = df.drop(columns = ['true_time', 'true_label']).rename(columns = {'label': 'event', 'time': 'duration'})
    else:
        return load_dsm(dataset, **kwargs)

    covariates = df.drop(['duration', 'event'], axis = 'columns')
    return StandardScaler().fit_transform(covariates.values).astype(float),\
           df['duration'].values.astype(float) + 1,\
           df['event'].values.astype(int),\
           covariates.columns