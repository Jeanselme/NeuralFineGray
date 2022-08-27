from dsm.dsm_api import DSMBase
from nfg.nfg_torch import NeuralFineGrayTorch
import nfg.losses as losses
from nfg.utilities import train_nfg

import torch
import numpy as np
from tqdm import tqdm

class NeuralFineGray(DSMBase):

  def __init__(self, cuda = torch.cuda.is_available(), mask = None, **params):
    self.params = params
    self.fitted = False
    self.cuda = cuda

  def _gen_torch_model(self, inputdim, optimizer, risks):
    model = NeuralFineGrayTorch(inputdim, **self.params,
                                     risks = risks,
                                     optimizer = optimizer).double()
    if self.cuda > 0:
      model = model.cuda()
    return model

  def fit(self, x, t, e, vsize = 0.15, val_data = None,
          optimizer = "Adam", random_state = 100, **args):
    processed_data = self._preprocess_training_data(x, t, e,
                                                   vsize, val_data,
                                                   random_state)
    x_train, t_train, e_train, x_val, t_val, e_val = processed_data

    maxrisk = int(np.nanmax(e_train.cpu().numpy()))
    model = self._gen_torch_model(x_train.size(1), optimizer, risks = maxrisk)
    model = train_nfg(model,
                         x_train, t_train, e_train,
                         x_val, t_val, e_val, cuda = self.cuda == 2,
                         **args)

    self.torch_model = model.eval()
    self.fitted = True
    return self    

  def compute_nll(self, x, t, e):
    if not self.fitted:
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `_eval_nll`.")
    processed_data = self._preprocess_training_data(x, t, e, 0, None, 0)
    _, _, _, x_val, t_val, e_val = processed_data
    if self.cuda == 2:
      x_val, t_val, e_val = x_val.cuda(), t_val.cuda(), e_val.cuda()

    loss = losses.total_loss(self.torch_model, x_val, t_val, e_val)
    return loss.item()

  def predict_survival(self, x, t, risk = None):
    x = self._preprocess_test_data(x)
    if not isinstance(t, list):
      t = [t]
    if self.fitted:
      scores = []
      for t_ in t:
        t_ = torch.DoubleTensor([t_] * len(x)).to(x.device)
        cumulative, _ = self.torch_model(x, t_)
        if risk is None:
          outcomes = torch.exp(- cumulative.sum(1))
          scores.append(outcomes.unsqueeze(1).detach().cpu().numpy())
        else:
          outcomes = torch.exp(- cumulative)
          scores.append(outcomes[:, int(risk) - 1].unsqueeze(1).detach().cpu().numpy())
      return np.concatenate(scores, axis = 1)
    else:
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `predict_survival`.")

  def predict_cif(self, x, t, risk = None, approx = 100):
    x = self._preprocess_test_data(x)
    if not isinstance(t, list):
      t = [t]
    if self.fitted:
      scores = []
      for t_ in t:
        cif = torch.zeros(len(x), self.torch_model.risks).to(x.device)
        for ti in np.linspace(0, t_, approx):
          ti = torch.DoubleTensor([ti] * len(x)).to(x.device)
          cumulative, intensity = self.torch_model(x, ti, gradient = True)
          survival = torch.exp(- cumulative.sum(dim = 1)).unsqueeze(1).repeat(1, self.torch_model.risks)
          cif += intensity.squeeze() * survival * t_ / approx
        if risk is None:
          scores.append(cif.detach().cpu().numpy())
        else:
          scores.append(cif[:, int(risk) - 1].unsqueeze(1).detach().cpu().numpy())
      return np.concatenate(scores, axis = 1)
    else:
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `predict_cif`.")