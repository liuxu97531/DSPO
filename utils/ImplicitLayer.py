import torch
from torch import nn
from .RBFFD_grad import RBFFD_grad

class ImplicitLayer(nn.Module):
    def __init__(self, x_nodes, x_nodes_total, u_exact, grad=True):
        super(ImplicitLayer, self).__init__()
        RBFFD = RBFFD_grad(order=5, n=100, phi='phs3')
        # RBFFD = RBFFD_grad(order=5, n=200, phi='phs3')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        f_tensor = lambda x: torch.tensor(x).float().to(device)
        if grad:
            self.weights1, self.weights2 = RBFFD.rbf_fd_grad_2d(x_nodes, x_nodes_total, u_exact)
            self.weights1, self.weights2 = f_tensor(self.weights1).T, f_tensor(self.weights2).T  # (N_power,n_obs)
        else:
            self.weights1, self.weights2 = None, None
        self.interp = f_tensor(RBFFD.rbf_fd_2d(x_nodes, x_nodes_total, u_exact)).T  # (N_power,n_obs)

    def forward(self):
        return self.weights1, self.weights2, self.interp
