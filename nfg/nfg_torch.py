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

  def __init__(self, inputdim, layers = [100, 100, 100], act = 'ReLU6', layers_surv = [100], act_surv = 'Tanh',
               risks = 1, dropout = 0., optimizer = "Adam"):
    super(NeuralFineGrayTorch, self).__init__()
    self.input_dim = inputdim
    self.risks = risks  # Competing risks
    self.dropout = dropout
    self.optimizer = optimizer

    self.embed = nn.Sequential(*create_representation(inputdim, layers + [inputdim], act, self.dropout)) # Assign each point to a cluster
    self.balance = nn.Sequential(*create_representation(inputdim, layers + [risks], act)) # Define balance between outcome (ensure sum < 1)
    self.outcome = nn.ModuleList(
                      [create_representation_positive(inputdim + 1, layers_surv + [1], act_surv) # Multihead (one for each outcome)
                  for _ in range(risks)]) 
    
    self.softlog = nn.LogSoftmax(dim = 1)

  def forward(self, x, horizon, gradient = False):
    x_rep = self.embed(x)

    # Commpute balance
    log_beta = self.softlog(self.balance(x_rep)).T
    
    # Compute outcomes 
    logbfs, logSs, logF = [], [], [] # Respecting notation in paper
    for risk, outcome_competing in zip(range(self.risks), self.outcome):
      # Through positive neural network
      tau_outcome = horizon.clone().detach().requires_grad_(gradient) # Copy with independent gradient
      zeros = outcome_competing(torch.cat((x_rep, torch.zeros_like(tau_outcome.unsqueeze(1))), 1)) # Outcome at time 0
      out = outcome_competing(torch.cat((x_rep, tau_outcome.unsqueeze(1)), 1)) # Outcome at time t

      # Compute the difference to ensure: S = 1 => F = 0 at time 0
      diff = (zeros - out).squeeze() # Because of softplus it is between 0 and -inf
      # F = beta * (1 - torch.exp(diff)) --- Ensure 0 < F < 1 (zeros == out -> diff = 0 -> F = 0, out goes to inf -> diff goes - inf -> F = 1)

      # Compute log survival log(1 - sum F) = log(1 - sum(beta (1 - exp[diff])))) = log(sum(beta * exp[diff] )) => **Use later log exp sum**
      logS = log_beta[risk] + diff
      logF.append(diff.unsqueeze(-1))
      logSs.append(logS.unsqueeze(-1))

      if gradient:
        # Expension of derivative dF/dt = beta d(1-exp(diff))/dt = beta exp(diff) * d(-diff)/dt => log(dF) = logbeta + diff + log(ddiff)
        ddiff = grad(out.mean(), tau_outcome, create_graph = True)[0] 
        logbf = log_beta[risk] + diff + torch.log(ddiff.clamp_(1e-8))
        logbfs.append(logbf.unsqueeze(1))

    logF = torch.cat(logF, -1)
    logSs = torch.cat(logSs, -1)
    logbfs = torch.cat(logbfs, -1) if gradient else None
    
    return logSs, logbfs, logF  
   
  def predict(self, x, horizon):
    _, _, logF = self.forward(x, horizon)
    return torch.exp(logF)

