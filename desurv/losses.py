import torch
import torch.nn as nn
import numpy as np

def total_loss(model, x, t, e, eps = 1e-8):
  # Require to be ordered by t
  ordering = torch.argsort(t)
  x_order, t_order, e_order = x[ordering], t[ordering], e[ordering]

  pred, xrep, ode = model.forward(x_order, t_order)

  # Likelihood error
  error = - torch.log(1 - pred[e_order == 0].sum(dim = 1) + eps).sum()
  for k in range(model.risks):
    ids = e_order == (k + 1)
    dudt = model.odenet.dudt(torch.cat((t_order[ids].unsqueeze(1), x_order[ids, :]), 1))

    error -= (torch.log(1 - ode[ids][:, k]**2 + eps) 
            + torch.log(dudt[:, k] + eps) 
            + torch.log(xrep[ids][:, k] + eps)).sum()

  return error / len(x)