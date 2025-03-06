from nfg.nfg_api import NeuralFineGray
from desurv.desurv_torch import DeSurvTorch
import desurv.losses as losses

import torch
import numpy as np

class DeSurv(NeuralFineGray):

  def _gen_torch_model(self, inputdim, optimizer, risks):
    self.loss = losses.total_loss
    model = DeSurvTorch(inputdim, **self.params,
                        risks = risks,
                        optimizer = optimizer).double()
    if self.cuda > 0:
      model = model.cuda()
    return model

  def predict_survival(self, x, t, risk = None):
    x = self._preprocess_test_data(x)
    if not isinstance(t, list):
      t = [t]
    if self.fitted:
      scores = []
      for t_ in t:
        t_ = self._normalise(torch.DoubleTensor([t_] * len(x))).to(x.device)
        pred, _, _, _ = self.torch_model(x, t_)
        if risk is None:
          scores.append(1 - pred.sum(1).unsqueeze(1).detach().cpu().numpy())
        else:
          scores.append(1 - pred[:, int(risk) - 1].unsqueeze(1).detach().cpu().numpy())
      return np.concatenate(scores, axis = 1)
    else:
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `predict_survival`.")