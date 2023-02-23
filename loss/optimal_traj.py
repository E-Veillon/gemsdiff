import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_mean

import crystallographic_graph


class OptimalTrajLoss(nn.Module):
    def __init__(
        self, center: bool = True, euclidian: bool = False, distance: str = "l1"
    ):
        super().__init__()

        self.center = center
        self.distance = distance
        self.euclidian = euclidian

        self.offset = nn.Parameter(
            torch.tensor(
                [
                    [-1, -1, -1],
                    [-1, -1, 0],
                    [-1, -1, 1],
                    [-1, 0, -1],
                    [-1, 0, 0],
                    [-1, 0, 1],
                    [-1, 1, -1],
                    [-1, 1, 0],
                    [-1, 1, 1],
                    [0, -1, -1],
                    [0, -1, 0],
                    [0, -1, 1],
                    [0, 0, -1],
                    [0, 0, 0],
                    [0, 0, 1],
                    [0, 1, -1],
                    [0, 1, 0],
                    [0, 1, 1],
                    [1, -1, -1],
                    [1, -1, 0],
                    [1, -1, 1],
                    [1, 0, -1],
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 1, -1],
                    [1, 1, 0],
                    [1, 1, 1],
                ],
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

    def forward(
        self,
        cell: torch.FloatTensor,
        x: torch.FloatTensor,
        x_tilde: torch.FloatTensor,
        x_traj: torch.FloatTensor,
        num_atoms: torch.LongTensor,
    ) -> torch.FloatTensor:

        struct_idx = torch.arange(cell.shape[0], device=cell.device)
        batch = struct_idx.repeat_interleave(num_atoms)

        euc_x_tilde = torch.einsum(cell[batch], [0, 1, 2], x_tilde, [0, 2], [0, 1])

        x_offset = x[:, None, :] + self.offset[None, :, :]
        euc_x = torch.einsum(cell[batch], [0, 2, 3], x_offset, [0, 1, 3], [0, 1, 2])

        traj = euc_x_tilde[:, None] - euc_x
        min_idx = traj.norm(dim=2).argmin(dim=1)

        idx = torch.arange(min_idx.shape[0], dtype=torch.long, device=min_idx.device)
        if self.euclidian:
            optimal_traj = traj[idx, min_idx]
        else:
            optimal_traj = x_tilde - x_offset[idx, min_idx]

        if self.center:
            optimal_traj -= scatter_mean(
                optimal_traj, batch, dim=0, dim_size=cell.shape[0]
            )[batch]

        if self.distance == "l1":
            return F.l1_loss(x_traj, optimal_traj)
        if self.distance == "mse":
            return F.mse_loss(x_traj, optimal_traj)
