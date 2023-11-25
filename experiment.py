from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, ShuffleSplit, ParameterSampler, train_test_split
import pandas as pd
import numpy as np
import pickle
import torch
import time
import os
import io

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location = 'cpu')
        else: 
            return super().find_class(module, name)

class ToyExperiment():

    def train(self, *args, cause_specific = False):
        print("Toy Experiment - Results already saved")

class Experiment():

    def __init__(self, hyper_grid = None, n_iter = 100, fold = None,
                k = 5, random_seed = 0, path = 'results', save = True, delete_log = False, times = 100):
        """
        Args:
            hyper_grid (Dict, optional): Dictionary of parameters to explore.
            n_iter (int, optional): Number of random grid search to perform. Defaults to 100.
            fold (int, optional): Fold to compute (this allows to parallelise computation). If None, starts from 0.
            k (int, optional): Number of split to use for the cross-validation. Defaults to 5.
            random_seed (int, optional): Random seed for reproducibility. Defaults to 0.
            path (str, optional): Path to save results and log. Defaults to 'results'.
            save (bool, optional): Should we save result and log. Defaults to True.
            delete_log (bool, optional): Should we delete the log after all training. Defaults to False.
            times (int, optional): Number of time points where to evaluates. Defaults to 100.
        """
        self.hyper_grid = list(ParameterSampler(hyper_grid, n_iter = n_iter, random_state = random_seed) if hyper_grid is not None else [{}])
        self.random_seed = random_seed
        self.k = k
        
        # Allows to reload a previous model
        self.all_fold = fold
        self.iter, self.fold = 0, 0 
        self.best_hyper = {}
        self.best_model = {}
        self.best_nll = None

        self.times = times

        self.path = path
        self.tosave = save
        self.delete_log = delete_log
        self.running_time = 0

    @classmethod
    def create(cls, hyper_grid = None, n_iter = 100, fold = None, k = 5,
                random_seed = 0, path = 'results', force = False, save = True, delete_log = False):
        if not(force):
            path = path if fold is None else path + '_{}'.format(fold)
            if os.path.isfile(path + '.csv'):
                return ToyExperiment()
            elif os.path.isfile(path + '.pickle'):
                print('Loading previous copy')
                try:
                    return cls.load(path+ '.pickle')
                except Exception as e:
                    print('ERROR: Reinitalizing object')
                    os.remove(path + '.pickle')
                    pass
                
        return cls(hyper_grid, n_iter, fold, k, random_seed, path, save, delete_log)

    @classmethod
    def load(cls, path):
        file = open(path, 'rb')
        if torch.cuda.is_available():
            return pickle.load(file)
        else:
            se = CPU_Unpickler(file).load()
            for i in se.best_model:
                if not isinstance(se.best_model[i], dict):
                    se.best_model[i].cuda = False
            return se

    @classmethod
    def merge(cls, hyper_grid = None, n_iter = 100, fold = None, k = 5,
            random_seed = 0, path = 'results', force = False, save = True, delete_log = False):
        merged = cls(hyper_grid, n_iter, fold, k, random_seed, path, save, delete_log)
        for i in range(5):
            path_i = path + '_{}.pickle'.format(i)
            if os.path.isfile(path_i):
                merged.best_model[i] = cls.load(path_i).best_model[i]
            else:
                print('Fold {} has not been computed yet'.format(i))
        merged.fold = 5 # Nothing to run
        return merged

    @classmethod
    def save(cls, obj):
        with open(obj.path + '.pickle', 'wb') as output:
            try:
                pickle.dump(obj, output)
            except Exception as e:
                print('Unable to save object')
                
    def save_results(self, x):
        predictions = []
        for i in self.best_model:
            index = self.fold_assignment[self.fold_assignment == i].index
            model = self.best_model[i]
            predictions.append(pd.concat([self._predict_(model, x[index], r, index) for r in self.risks], axis = 1))

        predictions = pd.concat(predictions, axis = 0).loc[self.fold_assignment.dropna().index]

        if self.tosave:
            fold_assignment = self.fold_assignment.copy().to_frame()
            fold_assignment.columns = pd.MultiIndex.from_product([['Use'], ['']])
            pd.concat([predictions, fold_assignment], axis = 1).to_csv(self.path + '.csv')

        if self.delete_log:
            os.remove(self.path + '.pickle')
        return predictions

    def train(self, x, t, e, cause_specific = False):
        """
            Cross validation model

            Args:
                x (Dataframe n * d): Observed covariates
                t (Dataframe n): Time of censoring or event
                e (Dataframe n): Event indicator

                cause_specific (bool): If model should be trained in cause specific setting

            Returns:
                (Dict, Dict): Dict of fitted model and Dict of observed performances
        """
        self.times = np.linspace(t.min(), t.max(), self.times) if isinstance(self.times, int) else self.times
        self.scaler = StandardScaler()
        x = self.scaler.fit_transform(x)

        self.risks = np.unique(e[e > 0])
        self.fold_assignment = pd.Series(np.nan, index = range(len(x)))
        if self.k == 1:
            kf = ShuffleSplit(n_splits = self.k, random_state = self.random_seed, test_size = 0.2)
        else:
            kf = StratifiedKFold(n_splits = self.k, random_state = self.random_seed, shuffle = True)

        # First initialization
        if self.best_nll is None:
            self.best_nll = np.inf
        for i, (train_index, test_index) in enumerate(kf.split(x, e)):
            self.fold_assignment[test_index] = i
            if i < self.fold: continue # When reload: start last point
            if not(self.all_fold is None) and (self.all_fold != i): continue
            print('Fold {}'.format(i))

            train_index, dev_index = train_test_split(train_index, test_size = 0.2, random_state = self.random_seed, stratify = e[train_index])
            dev_index, val_index   = train_test_split(dev_index,   test_size = 0.5, random_state = self.random_seed, stratify = e[dev_index])
            
            x_train, x_dev, x_val = x[train_index], x[dev_index], x[val_index]
            t_train, t_dev, t_val = t[train_index], t[dev_index], t[val_index]
            e_train, e_dev, e_val = e[train_index], e[dev_index], e[val_index]

            # Train on subset one domain
            ## Grid search best params
            for j, hyper in enumerate(self.hyper_grid):
                if j < self.iter: continue # When reload: start last point
                np.random.seed(self.random_seed)
                torch.manual_seed(self.random_seed)

                start_time = time.process_time()
                model = self._fit_(x_train, t_train, e_train, x_val, t_val, e_val, hyper.copy(), cause_specific = cause_specific)
                self.running_time += time.process_time() - start_time
                
                nll = self._nll_(model, x_dev, t_dev, e_dev, e_train, t_train)
                if nll < self.best_nll:
                    self.best_hyper[i] = hyper
                    self.best_model[i] = model
                    self.best_nll = nll

                self.iter = j + 1
                self.save(self)
            self.fold, self.iter = i + 1, 0
            self.best_nll = np.inf
            self.save(self)

        if self.all_fold is None:
            return self.save_results(x)

    def _fit_(self, *params):
        raise NotImplementedError()

    def _nll_(self, *params):
        raise NotImplementedError()

    def likelihood(self, x, t, e):
        x = self.scaler.transform(x)
        nll_fold = {}

        for i in self.best_model:
            index = self.fold_assignment[self.fold_assignment == i].index
            train = self.fold_assignment[self.fold_assignment != i].index
            model = self.best_model[i]
            if type(model) is dict:
                nll_fold[i] = np.mean([self._nll_(model[r], x[index], t[index], e[index] == r, e[train] == r, t[train]) for r in self.risks])
            else:
                nll_fold[i] = self._nll_(model, x[index], t[index], e[index], e[train], t[train])

        return nll_fold

class DSMExperiment(Experiment):

    def _fit_(self, x, t, e, x_val, t_val, e_val, hyperparameter, cause_specific):  
        from dsm import DeepSurvivalMachines

        epochs = hyperparameter.pop('epochs', 1000)
        batch = hyperparameter.pop('batch', 250)
        lr = hyperparameter.pop('learning_rate', 0.001)

        model = DeepSurvivalMachines(**hyperparameter, cuda = torch.cuda.is_available())
        model.fit(x, t, e, iters = epochs, batch_size = batch,
                learning_rate = lr, val_data = (x_val, t_val, e_val))
        
        return model

    def _nll_(self, model, x, t, e, *train):
        return model.compute_nll(x, t, e)

    def _predict_(self, model, x, r, index):
        return pd.DataFrame(model.predict_survival(x, self.times.tolist(), risk = r), columns = pd.MultiIndex.from_product([[r], self.times]), index = index)

class DeepHitExperiment(Experiment):
    """
        This class require a slightly more involved saving scheme to avoid a lambda error with pickle
        The models are removed at each save and reloaded before saving results 
    """

    @classmethod
    def load(cls, path):
        from pycox.models import DeepHitSingle, DeepHit
        file = open(path, 'rb')
        if torch.cuda.is_available():
            exp = pickle.load(file)
            for i in exp.best_model:
                if isinstance(exp.best_model[i], tuple):
                    net, cuts = exp.best_model[i]
                    exp.best_model[i] = DeepHit(net, duration_index = cuts) if len(exp.risks) > 1 \
                                    else DeepHitSingle(net, duration_index = cuts)
            return exp
        else:
            se = CPU_Unpickler(file).load()
            for i in se.best_model:
                if isinstance(se.best_model[i], tuple):
                    net, cuts = se.best_model[i]
                    se.best_model[i] = DeepHit(net, duration_index = cuts) if len(se.risks) > 1 \
                                    else DeepHitSingle(net, duration_index = cuts)
                    se.best_model[i].cuda = False
            return se

    @classmethod
    def save(cls, obj):
        from pycox.models import DeepHitSingle, DeepHit
        with open(obj.path + '.pickle', 'wb') as output:
            try:
                for i in obj.best_model:
                    # Split model and save components (error pickle otherwise)
                    if isinstance(obj.best_model[i], DeepHit) or isinstance(obj.best_model[i], DeepHitSingle):
                        obj.best_model[i] = (obj.best_model[i].net, obj.best_model[i].duration_index)
                pickle.dump(obj, output)
            except Exception as e:
                print('Unable to save object')

    def save_results(self, x):
        from pycox.models import DeepHitSingle, DeepHit

        # Reload models in memory
        for i in self.best_model:
            if isinstance(self.best_model[i], tuple):
                # Reload model
                net, cuts = self.best_model[i]
                self.best_model[i] = DeepHit(net, duration_index = cuts) if len(self.risks) > 1 \
                                else DeepHitSingle(net, duration_index = cuts)
        return super().save_results(x)

    def _fit_(self, x, t, e, x_val, t_val, e_val, hyperparameter, cause_specific): 
        from deephit.utils import CauseSpecificNet, tt, LabTransform
        from pycox.models import DeepHitSingle, DeepHit

        n = hyperparameter.pop('n', 15)
        nodes = hyperparameter.pop('nodes', [100])
        shared = hyperparameter.pop('shared', [100])
        epochs = hyperparameter.pop('epochs', 1000)
        batch = hyperparameter.pop('batch', 250)
        lr = hyperparameter.pop('learning_rate', 0.001)

        self.eval_times = np.linspace(0, t.max(), n)
        callbacks = [tt.callbacks.EarlyStopping()]
        num_risks = len(np.unique(e))- 1
        if  num_risks > 1:
            self.labtrans = LabTransform(self.eval_times.tolist())
            net = CauseSpecificNet(x.shape[1], shared, nodes, num_risks, self.labtrans.out_features, False)
            model = DeepHit(net, tt.optim.Adam, duration_index = self.labtrans.cuts)
        else:
            self.labtrans = DeepHitSingle.label_transform(self.eval_times.tolist())
            net = tt.practical.MLPVanilla(x.shape[1], shared + nodes, self.labtrans.out_features, False)
            model = DeepHitSingle(net, tt.optim.Adam, duration_index = self.labtrans.cuts)
        model.optimizer.set_lr(lr)
        model.fit(x.astype('float32'), self.labtrans.transform(t, e), batch_size = batch, epochs = epochs, 
                    callbacks = callbacks, val_data = (x_val.astype('float32'), self.labtrans.transform(t_val, e_val)))
        return model

    def _nll_(self, model, x, t, e, *train):
        return model.score_in_batches(x.astype('float32'), self.labtrans.transform(t, e))['loss']

    def _predict_(self, model, x, r, index):
        if len(self.risks) == 1:
            survival = model.predict_surv_df(x.astype('float32')).values
        else:
            survival = 1 - model.predict_cif(x.astype('float32'))[r - 1]

        # Interpolate at the point of evaluation
        survival = pd.DataFrame(survival, columns = index, index = self.eval_times)
        predictions = pd.DataFrame(np.nan, columns = index, index = self.times)
        survival = pd.concat([survival, predictions]).sort_index(kind = 'stable').bfill().ffill()
        survival = survival[~survival.index.duplicated(keep='first')]
        return survival.loc[self.times].set_index(pd.MultiIndex.from_product([[r], self.times])).T

class NFGExperiment(DSMExperiment):

    def __preprocess__(self, t, save = False):
        if save:
            self.max_t = t.max()
        return t / self.max_t

    def train(self, x, t, e, cause_specific = False):
        self.times = np.linspace(t.min(), t.max(), self.times) if isinstance(self.times, int) else self.times
        t_norm = self.__preprocess__(t, True)
        return super().train(x, t_norm, e, cause_specific)

    def _fit_(self, x, t, e, x_val, t_val, e_val, hyperparameter, cause_specific):  
        from nfg import NeuralFineGray

        epochs = hyperparameter.pop('epochs', 1000)
        batch = hyperparameter.pop('batch', 250)
        lr = hyperparameter.pop('learning_rate', 0.001)
        patience_max = hyperparameter.pop('patience_max', 3)

        model = NeuralFineGray(**hyperparameter, cause_specific = cause_specific)
        model.fit(x, t, e, n_iter = epochs, bs = batch, patience_max = patience_max,
                lr = lr, val_data = (x_val, t_val, e_val))
        
        return model

    def _predict_(self, model, x, r, index):
        return pd.DataFrame(model.predict_survival(x, self.__preprocess__(self.times).tolist(), r if model.torch_model.risks >= r else 1), columns = pd.MultiIndex.from_product([[r], self.times]), index = index)

    def likelihood(self, x, t, e):
        t_norm = self.__preprocess__(t)
        return super().likelihood(x, t_norm, e)

class DeSurvExperiment(NFGExperiment):

    def _fit_(self, x, t, e, x_val, t_val, e_val, hyperparameter, cause_specific):  
        from desurv import DeSurv

        epochs = hyperparameter.pop('epochs', 1000)
        batch = hyperparameter.pop('batch', 250)
        lr = hyperparameter.pop('learning_rate', 0.001)
        patience_max = hyperparameter.pop('patience_max', 3)

        model = DeSurv(**hyperparameter)
        model.fit(x, t, e, n_iter = epochs, bs = batch, patience_max = patience_max,
                lr = lr, val_data = (x_val, t_val, e_val))
        
        return model