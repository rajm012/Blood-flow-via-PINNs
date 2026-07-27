"""
model_3d.py
-----------
Network architecture and physics residuals for the 3D near-wall aneurysm
PINN, following Arzani, Wang & D'Souza (2021), Sec. II C 4 (test case 4:
3D aneurysm).

- Velocity components (u, v, w) and pressure (p) are each approximated by
  a SEPARATE fully connected network (paper, Sec. II B).
- Architecture used for the 3D case: 9 hidden layers, 200 neurons per
  layer (larger than the 128-neuron 2D cases), Swish activation.
- Steady, incompressible, nondimensional Navier-Stokes (3D):
      u*du/dx+v*du/dy+w*du/dz = -dp/dx + (1/Re)(d2u/dx2+d2u/dy2+d2u/dz2)
      u*dv/dx+v*dv/dy+w*dv/dz = -dp/dy + (1/Re)(d2v/dx2+d2v/dy2+d2v/dz2)
      u*dw/dx+v*dw/dy+w*dw/dz = -dp/dz + (1/Re)(d2w/dx2+d2w/dy2+d2w/dz2)
      du/dx + dv/dy + dw/dz = 0
  (all variables nondimensional; * dropped for readability)

WSS in 3D is a VECTOR (the 2D case's scalar "tangential derivative"
formula does not generalize directly): with u=0 exactly at a no-slip
wall, the viscous traction is t = (1/Re) * J . n  (J = velocity gradient
tensor, n = unit outward normal), and the wall shear stress vector is
the tangential part of that traction: WSS_vec = t - (t . n) n. Its
magnitude |WSS_vec| is what's comparable to the CFD scalar WSS export.
"""
import torch
import torch.nn as nn


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class MLP(nn.Module):
    """Fully connected network: input (x, y, z) -> single scalar output."""

    def __init__(self, in_dim=3, out_dim=1, hidden=200, n_hidden_layers=9):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), Swish()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden, hidden), Swish()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xyz):
        return self.net(xyz)


class PINN3D(nn.Module):
    """Wraps four separate MLPs (u, v, w, p) and exposes the 3D NS residuals."""

    def __init__(self, hidden=200, n_hidden_layers=9, Re=160.0):
        super().__init__()
        self.net_u = MLP(3, 1, hidden, n_hidden_layers)
        self.net_v = MLP(3, 1, hidden, n_hidden_layers)
        self.net_w = MLP(3, 1, hidden, n_hidden_layers)
        self.net_p = MLP(3, 1, hidden, n_hidden_layers)
        self.Re = Re

    def predict(self, x, y, z):
        xyz = torch.cat([x, y, z], dim=1)
        u = self.net_u(xyz)
        v = self.net_v(xyz)
        w = self.net_w(xyz)
        p = self.net_p(xyz)
        return u, v, w, p

    def _grad(self, f, wrt):
        return torch.autograd.grad(f, wrt, grad_outputs=torch.ones_like(f), create_graph=True)[0]

    def ns_residual(self, x, y, z):
        """Returns (r_mx, r_my, r_mz, r_cont). x,y,z must be leaf tensors
        with requires_grad_(True)."""
        u, v, w, p = self.predict(x, y, z)

        u_x, u_y, u_z = self._grad(u, x), self._grad(u, y), self._grad(u, z)
        v_x, v_y, v_z = self._grad(v, x), self._grad(v, y), self._grad(v, z)
        w_x, w_y, w_z = self._grad(w, x), self._grad(w, y), self._grad(w, z)
        p_x, p_y, p_z = self._grad(p, x), self._grad(p, y), self._grad(p, z)

        u_xx, u_yy, u_zz = self._grad(u_x, x), self._grad(u_y, y), self._grad(u_z, z)
        v_xx, v_yy, v_zz = self._grad(v_x, x), self._grad(v_y, y), self._grad(v_z, z)
        w_xx, w_yy, w_zz = self._grad(w_x, x), self._grad(w_y, y), self._grad(w_z, z)

        inv_Re = 1.0 / self.Re

        r_mx = u * u_x + v * u_y + w * u_z + p_x - inv_Re * (u_xx + u_yy + u_zz)
        r_my = u * v_x + v * v_y + w * v_z + p_y - inv_Re * (v_xx + v_yy + v_zz)
        r_mz = u * w_x + v * w_y + w * w_z + p_z - inv_Re * (w_xx + w_yy + w_zz)
        r_cont = u_x + v_y + w_z

        return r_mx, r_my, r_mz, r_cont

    def velocity_gradients(self, x, y, z):
        """Returns u,v,w,p and the full 3x3 velocity gradient tensor
        (needed for 3D WSS in evaluate_3d.py)."""
        u, v, w, p = self.predict(x, y, z)
        u_x, u_y, u_z = self._grad(u, x), self._grad(u, y), self._grad(u, z)
        v_x, v_y, v_z = self._grad(v, x), self._grad(v, y), self._grad(v, z)
        w_x, w_y, w_z = self._grad(w, x), self._grad(w, y), self._grad(w, z)
        return u, v, w, p, (u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y, w_z)
