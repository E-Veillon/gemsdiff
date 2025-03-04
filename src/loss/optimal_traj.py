"""Loss function comparing the difference between optimal and predicted atoms displacement trajectories inside the 3D torus space."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import product


class OptimalTrajLoss(nn.Module):
    """
    Loss function comparing the difference between optimal and predicted
    atoms displacement trajectories inside the 3D torus space.

    Parameters:
        center (bool):      TODO: Unused argument ? Defaults to True.

        euclidian (bool):   Whether to compute euclidian representations of trajectories
                            before loss measurement.

        distance (str):     distance formula to use for comparison.
                            Supports "l1" and "mse" distances.

    Returns:
        torch.Tensor: Tensor containing loss values.
    """
    def __init__(
        self, center: bool = True, euclidian: bool = False, distance: str = "l1"
    ):
        """Init."""
        super().__init__()

        self.center = center
        self.distance = distance
        self.euclidian = euclidian

        self.offset = nn.Parameter( # represents the possible shifts to neighbors unit cells
            torch.tensor(list(product((-1, 0, 1), repeat=3)), dtype=torch.float32),
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
        """
        Compute optimal atoms displacement trajectory then compare it to model predicted one.

        Parameters:
            cell (torch.FloatTensor):       Tensor concatenating structures unit cell parameters.

            x (torch.FloatTensor):          Tensor concatenating original structures atomic
                                            positions from the dataset.

            x_tilde (torch.FloatTensor):    Tensor concatenating atomic positions of randomly noised
                                            structures.
                                            
            x_traj (torch.FloatTensor):     Tensor of the predicted displacement trajectories of atoms
                                            by the model from the x_tilde noisy positions.

            num_atoms (torch.FloatTensor):  1-D Tensor concatenating structures number of atoms.

        Returns:
            torch.FloatTensor: Tensor of element-wise MAE or MSE loss values.
        """
        # translate all positions in neighbor unit cells inside main unit cell.
        x = x % 1.0
        x_tilde = x_tilde % 1.0

        # Build structures indices and span them to the size of num_atoms to assign atoms to the right structures
        struct_idx = torch.arange(cell.shape[0], device=cell.device)
        batch = struct_idx.repeat_interleave(num_atoms)

        # Compute actual optimal trajectory between noisy positions x_tilde and true positions x
        euc_x_tilde = torch.einsum(cell[batch], [0, 1, 2], x_tilde, [0, 2], [0, 1])

        x_offset = x[:, None, :] + self.offset[None, :, :]
        euc_x = torch.einsum(cell[batch], [0, 2, 3], x_offset, [0, 1, 3], [0, 1, 2])

        traj = euc_x_tilde[:, None] - euc_x
        min_idx = traj.norm(dim=2).argmin(dim=1)

        # Eventually adapt trajectories representations
        idx = torch.arange(min_idx.shape[0], dtype=torch.long, device=min_idx.device)

        if self.euclidian:
            optimal_traj = -traj[idx, min_idx]
            x_traj = torch.bmm(cell[batch], x_traj.unsqueeze(2)).squeeze(2)
        else:
            optimal_traj = -(x_tilde - x_offset[idx, min_idx])

        # Compute the difference between predicted and optimal trajectories
        if self.distance == "l1":
            return (x_traj - optimal_traj).abs().mean()
        if self.distance == "mse":
            return F.mse_loss(x_traj, optimal_traj)
