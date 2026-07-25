"""
model.py
--------
Network architecture and physics residuals for the 2D steady stenosis PINN,
following Arzani, Wang & D'Souza (2021), Phys. Fluids 33, 071905, Sec. II B
and Sec. II C 2 (test case 2: 2D stenosis).

- Velocity components (u, v) and pressure (p) are each approximated by a
  SEPARATE fully connected network (paper, Sec. II B): "Velocity vector
  components and the pressure are each approximated with a separate NN."
- Architecture used for the 2D stenosis case: 9 hidden layers, 128 neurons
  per layer, Swish activation ( x * sigmoid(x) ).
- Steady, incompressible, nondimensional Navier-Stokes:
      u* du*/dx* + v* du*/dy* = -dp*/dx* + (1/Re)( d2u*/dx*2 + d2u*/dy*2 )
      u* dv*/dx* + v* dv*/dy* = -dp*/dy* + (1/Re)( d2v*/dx*2 + d2v*/dy*2 )
      du*/dx* + dv*/dy* = 0
"""
import torch
import torch.nn as nn


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class MLP(nn.Module):
    """Fully connected network: input (x, y) -> single scalar output."""

    def __init__(self, in_dim=2, out_dim=1, hidden=128, n_hidden_layers=9):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), Swish()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden, hidden), Swish()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

        # Xavier init tends to play nicer with Swish/tanh-like activations
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xy):
        return self.net(xy)


class PINN(nn.Module):
    """Wraps three separate MLPs (u, v, p) and exposes the NS residuals."""

    def __init__(self, hidden=128, n_hidden_layers=9, Re=75.0):
        super().__init__()
        self.net_u = MLP(2, 1, hidden, n_hidden_layers)
        self.net_v = MLP(2, 1, hidden, n_hidden_layers)
        self.net_p = MLP(2, 1, hidden, n_hidden_layers)
        self.Re = Re

    def predict(self, x, y):
        xy = torch.cat([x, y], dim=1)
        u = self.net_u(xy)
        v = self.net_v(xy)
        p = self.net_p(xy)
        return u, v, p

    def ns_residual(self, x, y):
        """Returns (r_mom_x, r_mom_y, r_cont) residuals at points (x, y).
        x, y must be leaf tensors with requires_grad_(True)."""
        u, v, p = self.predict(x, y)

        grad_outputs = torch.ones_like(u)

        u_x = torch.autograd.grad(u, x, grad_outputs=grad_outputs, create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=grad_outputs, create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=grad_outputs, create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=grad_outputs, create_graph=True)[0]
        p_x = torch.autograd.grad(p, x, grad_outputs=grad_outputs, create_graph=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=grad_outputs, create_graph=True)[0]

        u_xx = torch.autograd.grad(u_x, x, grad_outputs=grad_outputs, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=grad_outputs, create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=grad_outputs, create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=grad_outputs, create_graph=True)[0]

        inv_Re = 1.0 / self.Re

        r_mom_x = u * u_x + v * u_y + p_x - inv_Re * (u_xx + u_yy)
        r_mom_y = u * v_x + v * v_y + p_y - inv_Re * (v_xx + v_yy)
        r_cont = u_x + v_y

        return r_mom_x, r_mom_y, r_cont

    def velocity_gradients(self, x, y):
        """Returns u, v and their first derivatives (needed for WSS in eval)."""
        u, v, p = self.predict(x, y)
        grad_outputs = torch.ones_like(u)
        u_x = torch.autograd.grad(u, x, grad_outputs=grad_outputs, create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=grad_outputs, create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=grad_outputs, create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=grad_outputs, create_graph=True)[0]
        return u, v, p, u_x, u_y, v_x, v_y
