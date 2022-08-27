import torch
import torch.nn as nn
import numpy as np

def total_loss(model, x, t, e, eps = 1e-8):
  # Go through network
  cumulative, intensity = model.forward(x, t, gradient = True)
  with torch.no_grad():
    intensity.clamp_(eps)

  # Likelihood error
  error = cumulative.sum(1).sum() # Sum over the different risks and then across patient
  for k in range(model.risks):
      i = intensity[e == (k + 1)][:, k]
      error -= torch.log(i).sum() # Sum over patients with this risk

  return error / len(x)