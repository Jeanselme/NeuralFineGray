import torch
import torch.nn as nn
import numpy as np

def total_loss(model, x, t, e, eps = 1e-8):

  # Go through network
  _, log_cifs, cifs_grad = model.forward(x, t, gradient = True)

  # Likelihood error
  cum = - np.log(model.risks) + log_cifs 

  # Censored patients
  error = torch.logsumexp(cum[e == 0], dim = [1]).sum() # Sum over the risk and then across patient

  # Uncensored
  for k in range(model.risks):
    error += torch.log(cifs_grad[e == (k + 1)][:, k].clamp_(eps)).sum()

  return - error / len(x)