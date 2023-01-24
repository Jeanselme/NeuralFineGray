import torch
import torch.nn as nn
import numpy as np

def total_loss(model, x, t, e):
  # Go through network
  log_sr, log_hr, log_b = model.forward(x, t, gradient = True)

  # Compute competing risks
  log_balance_sr = log_b + log_sr
  log_balance_derivative = log_b + log_sr + log_hr

  # Likelihood error
  error = - torch.logsumexp(log_balance_sr[e == 0], dim = 1).sum() # Sum over the different risks and then across patient
  for k in range(model.risks):
      error -= log_balance_derivative[e == (k + 1)][:, k].sum() # Sum over patients with this risk

  return error / len(x)

def total_loss_cs(model, x, t, e):
  # Go through network
  log_sr, log_hr, _ = model.forward(x, t, gradient = True)

  # Likelihood error
  error = 0
  for k in range(model.risks):
    error += log_sr[e != k + 1][:, k].sum()
    error += log_hr[e == k + 1][:, k].sum()

  return - error / len(x)