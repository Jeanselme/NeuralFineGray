import torch
import torch.nn as nn
import numpy as np

def total_loss(model, x, t, e):

  # Go through network
  logSs, logbfs, _ = model.forward(x, t, gradient = True)

  # Censored patients
  error = torch.logsumexp(logSs[e == 0], dim = 1).sum() # Sum over the risk and then across patient

  # Uncensored
  for k in range(model.risks):
    error += logbfs[e == (k + 1)][:, k].sum()

  return - error / len(x)