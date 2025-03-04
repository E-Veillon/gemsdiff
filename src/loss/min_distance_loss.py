"""TODO: Module description."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_mean

from itertools import product


class MinDistanceLoss(nn.Module):
    """
    TODO: class description.

    Parameters:
        center (bool):  TODO. Defaults to True.

    Returns:
        torch.Tensor: Tensor of the loss values.
    """
    def __init__(self, center: bool = True):
        """Init."""
        super().__init__()

        self.center = center

        self.offset = nn.Parameter( # represents the possible shifts to neighbors unit cells
            torch.tensor(list(product((-1, 0, 1), repeat=3)), dtype=torch.float32),
            requires_grad=False,
        )


    def forward(
        self,
        cell: torch.FloatTensor,
        x: torch.FloatTensor,
        x_tilde: torch.FloatTensor,
        num_atoms: torch.LongTensor,
    ) -> torch.FloatTensor:
        """
        TODO: method description.

        Parameters:
            cell (torch.FloatTensor):       Tensor concatenating structures unit cell parameters.

            x (torch.FloatTensor):          Tensor concatenating original structures atomic
                                            positions from the dataset.

            x_tilde (torch.FloatTensor):    Tensor concatenating atomic positions of randomly
                                            noised structures.

            num_atoms (torch.FloatTensor):  1-D Tensor concatenating structures number of atoms.

        Returns:
            torch.FloatTensor: Tensor of element-wise MAE loss values.
        """
        struct_idx = torch.arange(cell.shape[0], device=cell.device)
        batch = struct_idx.repeat_interleave(num_atoms)

        euc_x_tilde = torch.einsum(cell[batch], [0, 1, 2], x_tilde, [0, 2], [0, 1])

        euc_x = torch.einsum(
            cell[batch],
            [0, 2, 3],
            x[:, None, :] + self.offset[None, :, :],
            [0, 1, 3],
            [0, 1, 2],
        )

        min_idx = (euc_x_tilde[:, None] - euc_x).norm(dim=2).argmin(dim=1)

        if self.center:
            center = scatter_mean(
                x + self.offset[min_idx] - x_tilde, batch, dim=0, dim_size=cell.shape[0]
            )

            return F.l1_loss(x_tilde, x + self.offset[min_idx] - center[batch])
        else:
            return F.l1_loss(x_tilde, x + self.offset[min_idx])
