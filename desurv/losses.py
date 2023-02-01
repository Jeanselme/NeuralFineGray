import torch
import torch.nn as nn
import numpy as np

def total_loss(model, x, t, e, eps = 1e-10):
  pred, balance, ode, embed = model.forward(x, t)

  # Likelihood error
  error = - torch.log(1 - pred[e == 0].sum(dim = 1) + eps).sum()
  for k in range(model.risks):
    ids = (e == (k + 1))
    derivative = model.odenet.f(torch.cat((t[ids].unsqueeze(1), embed[ids]), 1))
    error -= (torch.log(1 - ode[ids][:, k] ** 2 + eps) 
            + torch.log(derivative[:, k] + eps) 
            + torch.log(balance[ids][:, k] + eps)).sum()

  return error / len(x)