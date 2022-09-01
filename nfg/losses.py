import torch
import torch.nn as nn
import numpy as np

def total_loss(model, x, t, e, eps = 1e-8):
  # Go through network
  log_X, ll_obs, _ = model.forward(x, t, gradient = True)

  # Likelihood error
  error = - torch.logsumexp(log_X[e == 0], dim = 1).sum() # Sum over the different risks and then across patient
  for k in range(model.risks):
      error -= ll_obs[e == (k + 1)][:, k].sum() # Sum over patients with this risk

  return error / len(x)