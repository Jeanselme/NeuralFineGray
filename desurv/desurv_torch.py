from nfg.nfg_torch import *

class CondODENet(nn.Module):
    """
      Code extracted from https://github.com/djdanks/DeSurv
    """
    def __init__(self, cov_dim, layers, output_dim,
                 act = "ReLU", n = 15):
        super().__init__()
        self.output_dim = output_dim

        self.dudt = nn.Sequential(*create_representation_positive(cov_dim + 1, layers + [output_dim], act))
        self.n = n
        u_n, w_n = np.polynomial.legendre.leggauss(n)
        self.u_n = nn.Parameter(torch.tensor(u_n, dtype = torch.float32)[None,:], requires_grad = False)
        self.w_n = nn.Parameter(torch.tensor(w_n, dtype = torch.float32)[None,:], requires_grad = False)

    def mapping(self, x_):
        t = x_[:,0][:,None]
        x = x_[:,1:]
        tau = torch.matmul(t/2, 1+self.u_n) # N x n
        tau_ = torch.flatten(tau)[:,None] # Nn x 1. Think of as N n-dim vectors stacked on top of each other
        reppedx = torch.repeat_interleave(x, torch.tensor([self.n]*t.shape[0], dtype = torch.long).to(x_.device), dim=0)
        taux = torch.cat((tau_, reppedx),1) # Nn x (d+1)
        f_n = self.dudt(taux).reshape((*tau.shape, self.output_dim)) # N x n x d_out
        pred = t / 2 * (self.w_n[:, :, None] * f_n).sum(dim=1)
        return pred

    def forward(self, x_):
        return torch.tanh(self.mapping(x_))


class DeSurvTorch(nn.Module):

  def __init__(self, inputdim, layers = [100, 100, 100], act = 'ReLU', layers_surv = [100],
               risks = 1, dropout = 0., optimizer = "Adam"):
    super(DeSurvTorch, self).__init__()
    self.input_dim = inputdim
    self.risks = risks  # Competing risks
    self.dropout = dropout
    self.optimizer = optimizer

    self.embed = nn.Sequential(*create_representation(inputdim, layers + [risks], act, self.dropout)) # Transform input into risk specific
    self.log = nn.Softmax(dim = 1)
    self.odenet = CondODENet(inputdim, layers_surv, risks)

  def forward(self, x, horizon):
    x_rep = self.log(self.embed(x))
    ode_rep = self.odenet(torch.cat((horizon.unsqueeze(1), x), 1))
  
    return x_rep*ode_rep, x_rep, ode_rep