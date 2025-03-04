"""Compute MAE loss value between true and predicted edges between atoms."""
import torch
import torch.nn as nn
import torch.nn.functional as F

import crystallographic_graph


class RelativeLoss(nn.Module):
    """
    Compute MAE loss value between true and predicted edges between atoms.

    Parameters:
        knn (int):  Number k of nearest neighbors atoms to consider in the
                    loss computation. Defaults to 4.
    """
    def __init__(self, knn: int = 4):
        """
        Compute MAE loss value between true and predicted edges between atoms.

        Parameters:
            knn (int):  Number k of nearest neighbors atoms to consider in the
                        loss computation. Defaults to 4.
        """
        super().__init__()

        self.knn = knn

    def forward(
        self,
        cell: torch.FloatTensor,
        x: torch.FloatTensor,
        x_tilde: torch.FloatTensor,
        num_atoms: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Compute MAE loss value between true and predicted edges between atoms.

        Parameters:
            cell (torch.FloatTensor):       Tensor concatenating structures unit cell parameters.

            x (torch.FloatTensor):          Tensor concatenating original structures atomic
                                            positions.

            x_tilde (torch.FloatTensor):    Tensor concatenating atomic positions of structures
                                            denoised by model.

            num_atoms (torch.FloatTensor):  1-D Tensor concatenating structures number of atoms.

        Returns:
            torch.FloatTensor: Tensor of element-wise MAE loss values.
        """
        edges = crystallographic_graph.make_graph(cell, x, num_atoms, knn=self.knn)
        e_ij = x[edges.dst] + edges.cell - x[edges.src]
        e_tilde_ij = x_tilde[edges.dst] + edges.cell - x_tilde[edges.src]

        return F.l1_loss(e_tilde_ij, e_ij), (e_tilde_ij - e_ij).abs()
