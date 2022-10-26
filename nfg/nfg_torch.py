from torch.autograd import grad
import numpy as np
import torch.nn as nn
import torch

# All of this as dependence
class PositiveLinear(nn.Module):
  def __init__(self, in_features, out_features, bias = False):
    super(PositiveLinear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.log_weight = nn.Parameter(torch.Tensor(out_features, in_features))
    if bias:
      self.bias = nn.Parameter(torch.Tensor(out_features))
    else:
      self.register_parameter('bias', None)
    self.reset_parameters()

  def reset_parameters(self):
    nn.init.xavier_uniform_(self.log_weight)
    if self.bias is not None:
      fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.log_weight)
      bound = np.sqrt(1 / np.sqrt(fan_in))
      nn.init.uniform_(self.bias, -bound, bound)
    self.log_weight.data.abs_().sqrt_()

  def forward(self, input):
    if self.bias is not None:
      return nn.functional.linear(input, self.log_weight ** 2, self.bias)
    else:
      return nn.functional.linear(input, self.log_weight ** 2)


def create_representation_positive(inputdim, layers, activation, dropout = 0):
  modules = []
  if activation == 'ReLU6':
    act = nn.ReLU6()
  elif activation == 'ReLU':
    act = nn.ReLU()
  elif activation == 'Tanh':
    act = nn.Tanh()
  else:
    raise ValueError("Unknown {} activation".format(activation))
  
  prevdim = inputdim
  for hidden in layers:
    modules.append(PositiveLinear(prevdim, hidden, bias = True))
    if dropout > 0:
      modules.append(nn.Dropout(p = dropout))
    modules.append(act)
    prevdim = hidden

  # Need all values positive
  modules[-1] = nn.Softplus()

  return nn.Sequential(*modules)

def create_representation(inputdim, layers, activation, dropout = 0.5):
  if activation == 'ReLU6':
    act = nn.ReLU6()
  elif activation == 'ReLU':
    act = nn.ReLU()
  elif activation == 'Tanh':
    act = nn.Tanh()

  modules = []
  prevdim = inputdim

  for hidden in layers:
    modules.append(nn.Linear(prevdim, hidden, bias = True))
    if dropout > 0:
      modules.append(nn.Dropout(p = dropout))
    modules.append(act)
    prevdim = hidden
  
  return modules

class NeuralFineGrayTorch(nn.Module):

  def __init__(self, inputdim, layers = [100, 100, 100], act = 'ReLU', layers_surv = [100],
               risks = 1, dropout = 0., optimizer = "Adam"):
    super(NeuralFineGrayTorch, self).__init__()
    self.input_dim = inputdim
    self.risks = risks  # Competing risks
    self.dropout = dropout
    self.optimizer = optimizer

    self.embed = nn.Sequential(*create_representation(inputdim, layers + [inputdim], act, self.dropout)) # Assign each point to a cluster
    self.balance = nn.Sequential(*create_representation(inputdim, layers + [risks], act, self.dropout)) # Define balance between outcome (ensure sum < 1)
    self.outcome = nn.ModuleList(
                      [create_representation_positive(inputdim + 1, layers_surv + [1], 'Tanh') # Multihead (one for each outcome)
                  for _ in range(risks)]) 
    self.softlog = nn.LogSoftmax(dim = 1)

  def forward(self, x, horizon, gradient = False):
    x_rep = self.embed(x)
    log_beta = self.softlog(self.balance(x_rep)) # Balance

    # Compute cumulative hazard function
    log_X, ll_obs = [], []
    for risk, outcome_competing in zip(range(self.risks), self.outcome):
      tau_outcome = horizon.clone().detach().requires_grad_(gradient) # Copy with independent gradient
      outcome = outcome_competing(torch.cat((x_rep, tau_outcome.unsqueeze(1)), 1))
      N_r = (outcome_competing(torch.cat((x_rep, torch.zeros_like(tau_outcome.unsqueeze(1))), 1)) - outcome).squeeze()
      log_X.append((log_beta[:, risk] + N_r).unsqueeze(1))

      if gradient:
        derivative = grad(outcome.sum(), tau_outcome, create_graph = True)[0]
        ll_obs.append((log_beta[:, risk] + N_r + torch.log(derivative.clamp_(1e-8))).unsqueeze(1))

    log_X = torch.cat(log_X, -1)
    ll_obs = torch.cat(ll_obs, -1) if gradient else None

    return log_X, ll_obs, log_beta