from dsm.datasets import load_dataset as load_dsm
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from pycox import datasets
import pandas as pd
import numpy as np

EPS = 1e-8

def load_dataset(dataset='SUPPORT', normalize = True, **kwargs):
    if dataset == 'GBSG':
        df = datasets.gbsg.read_df()
    elif dataset == 'METABRIC':
        df = datasets.metabric.read_df()
        df = df.rename(columns = {'x0': 'MKI67', 'x1': 'EGFR', 'x2': 'PGR', 'x3': 'ERBB2', 
                                  'x4': 'Hormone', 'x5': 'Radiotherapy', 'x6': 'Chemotherapy', 'x7': 'ER-positive', 
                                  'x8': 'Age at diagnosis'})
        df['duration'] += EPS # Avoid problem of the minimum value 0
    elif dataset == 'SYNTHETIC':
        df = datasets.rr_nl_nhp.read_df()
        df = df.drop([c for c in df.columns if 'true' in c], axis = 'columns')
    elif dataset == 'SEER':
        df = pd.read_csv('../data/export.csv')
        df = process_seer(df)
        df['duration'] += EPS # Avoid problem of the minimum value 0
    elif dataset == 'SYNTHETIC_COMPETING':
        df = pd.read_csv('https://raw.githubusercontent.com/chl8856/DeepHit/master/sample%20data/SYNTHETIC/synthetic_comprisk.csv')
        df = df.drop(columns = ['true_time', 'true_label']).rename(columns = {'label': 'event', 'time': 'duration'})
        df['duration'] += EPS # Avoid problem of the minimum value 0
    else:
        return load_dsm(dataset, normalize = normalize, **kwargs)

    covariates = df.drop(['duration', 'event'], axis = 'columns')
    return StandardScaler().fit_transform(covariates.values).astype(float) if normalize else covariates.values.astype(float),\
           df['duration'].values.astype(float),\
           df['event'].values.astype(int),\
           covariates.columns

def process_seer(df):
    # Remove multiple visits
    df = df.groupby('Patient ID').first().drop(columns= ['Site recode ICD-O-3/WHO 2008'])

    # Encode using dictionary to remove missing data
    df["RX Summ--Surg Prim Site (1998+)"].replace('126', np.nan, inplace = True)
    df["Sequence number"].replace(['88', '99'], np.nan, inplace = True)
    df["Regional nodes positive (1988+)"].replace(['95', '96', '97', '98', '99', '126'], np.nan, inplace = True)
    df["Regional nodes examined (1988+)"].replace(['95', '96', '97', '98', '99', '126'], np.nan, inplace = True)
    df = df.replace(['Blank(s)', 'Unknown'], np.nan).rename(columns = {"Survival months": "duration"})

    # Remove patients without survival time
    df = df[~df.duration.isna()]

    # Outcome 
    df['event'] = df["Vital status recode (study cutoff used)"] == 'Dead' # Death 
    df['event'].loc[(df["COD to site recode"] == "Diseases of Heart") & (df["SEER cause-specific death classification"] == "Alive or dead of other cause")] = 2 # CVD 

    df = df.drop(columns = ["COD to site recode"])

    # Imput and encode categorical
    ## Categorical
    categorical_col = ["Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)", "Laterality", 
        "Diagnostic Confirmation", "Histology recode - broad groupings", "Chemotherapy recode (yes, no/unk)",
        "Radiation recode", "ER Status Recode Breast Cancer (1990+)", "PR Status Recode Breast Cancer (1990+)",
        "Histologic Type ICD-O-3", "ICD-O-3 Hist/behav, malignant", "Sequence number", "RX Summ--Surg Prim Site (1998+)",
        "CS extension (2004-2015)", "CS lymph nodes (2004-2015)", "CS mets at dx (2004-2015)", "Year of diagnosis",
        "Origin recode NHIA (Hispanic, Non-Hisp)"]
    ordinal_col = ["Age recode with <1 year olds", "Grade"]

    imputer = SimpleImputer(strategy='most_frequent')
    df_cat = pd.DataFrame(imputer.fit_transform(df[categorical_col + ordinal_col]), columns = categorical_col + ordinal_col, index = df.index)

    df_cat = pd.get_dummies(df_cat, drop_first = True, columns = categorical_col + ordinal_col)
    
    ## Numerical
    numerical_col = ["Total number of in situ/malignant tumors for patient", "Total number of benign/borderline tumors for patient",
          "CS tumor size (2004-2015)", "Regional nodes examined (1988+)", "Regional nodes positive (1988+)"]
    imputer = SimpleImputer(strategy='mean')
    df_num = pd.DataFrame(imputer.fit_transform(df[numerical_col].astype(float)), columns = numerical_col, index = df.index)

    return pd.concat([df_cat, df_num, df[['duration', 'event']]], axis = 1)
    
    