import torch
import torch.nn as nn
import torch.nn.functional as F


class Grad(nn.Module):
    def __init__(self):
        super().__init__()

        self.I = nn.Parameter(torch.eye(3), requires_grad=False)
        self.K = nn.Parameter(
            torch.tensor(
                [
                    [[0, 0, 0], [0, 0, 1], [0, -1, 0]],
                    [[0, 0, -1], [0, 0, 0], [1, 0, 0]],
                    [[0, 1, 0], [-1, 0, 0], [0, 0, 0]],
                ],
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

    def jacobian_atan2(self, y, x):
        diff_x = -y / (x**2 + y**2)
        diff_y = x / (x**2 + y**2)

        return diff_y, diff_x

    def jacobian_dot(self, x, y):
        return y.clone(), x.clone()

    def jacobian_norm(self, x):
        return F.normalize(x, dim=1)

    def jacobian_cross_norm(self, x, y):
        diff_cross_x = (self.K[None] * y[:, None, None, :]).sum(dim=3)
        diff_cross_y = -(self.K[None] * x[:, None, None, :]).sum(dim=3)

        diff_norm = self.jacobian_norm(torch.cross(x, y))
        diff_x = torch.bmm(diff_norm.unsqueeze(1), diff_cross_x).squeeze(1)
        diff_y = torch.bmm(diff_norm.unsqueeze(1), diff_cross_y).squeeze(1)

        return diff_x, diff_y

    def jacobian_m(self, u):
        diff_m = self.I[None, :, :, None] * u[:, None, None, :]
        return diff_m

    def jacobian_mu(self, m, u):
        diff_m = self.I[None, :, :, None] * u[:, None, None, :]
        diff_u = m.clone()
        return diff_m, diff_u

    def jacobian_angle_vector(self, u, v):
        diff_atan2_y, diff_atan2_x = self.jacobian_atan2(
            torch.cross(u, v).norm(dim=1), (u * v).sum(dim=1)
        )

        diff_cross_norm_u, diff_cross_norm_v = self.jacobian_cross_norm(u, v)
        diff_dot_u, diff_dot_v = self.jacobian_dot(u, v)

        diff_u = (
            diff_atan2_y[:, None] * diff_cross_norm_u
            + diff_atan2_x[:, None] * diff_dot_u
        )
        diff_v = (
            diff_atan2_y[:, None] * diff_cross_norm_v
            + diff_atan2_x[:, None] * diff_dot_v
        )

        return diff_u, diff_v

    def grad_distance(self, rho, x_ij, g=None):
        if g is None:
            rho_prime = rho
        else:
            rho_prime = torch.bmm(g, rho)

        u = torch.bmm(rho_prime, x_ij.unsqueeze(2)).squeeze(2)

        diff_u = self.jacobian_norm(u)

        diff_g_u = self.jacobian_m(torch.bmm(rho, x_ij.unsqueeze(2)).squeeze(2))

        diff_g = torch.einsum("bi,bijk->bjk", diff_u, diff_g_u)

        return diff_g

    def grad_angle(self, rho, x_ij, x_ik, g=None):
        if g is None:
            rho_prime = rho
        else:
            rho_prime = torch.bmm(g, rho)

        u = torch.bmm(rho_prime, x_ij.unsqueeze(2)).squeeze(2)
        v = torch.bmm(rho_prime, x_ik.unsqueeze(2)).squeeze(2)

        diff_u, diff_v = self.jacobian_angle_vector(u, v)

        diff_g_u = self.jacobian_m(torch.bmm(rho, x_ij.unsqueeze(2)).squeeze(2))
        diff_g_v = self.jacobian_m(torch.bmm(rho, x_ik.unsqueeze(2)).squeeze(2))

        diff_g = torch.einsum("bi,bijk->bjk", diff_u, diff_g_u) + torch.einsum(
            "bi,bijk->bjk", diff_v, diff_g_v
        )

        return diff_g

    @property
    def triplets_dim(self):
        return 3

    def forward(
        self,
        cell: torch.FloatTensor,
        batch_triplets: torch.LongTensor,
        e_ij: torch.FloatTensor,
        e_ik: torch.FloatTensor,
    ):
        return torch.stack(
            (
                self.grad_distance(cell[batch_triplets], e_ij),
                self.grad_distance(cell[batch_triplets], e_ik),
                self.grad_angle(cell[batch_triplets], e_ij, e_ik),
            ),
            dim=1,
        )
