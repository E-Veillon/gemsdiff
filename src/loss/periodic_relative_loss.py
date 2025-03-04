"""Compute MAE loss value between true and predicted edges between atoms, taking account of Periodic Boundary Conditions inside 3-D torus space."""
import torch
import torch.nn as nn
import torch.nn.functional as F

import crystallographic_graph

from .min_distance_loss import MinDistanceLoss


class PeriodicRelativeLoss(nn.Module):
    """
    Compute MAE loss value between true and predicted edges between atoms,
    taking account of Periodic Boundary Conditions inside 3-D torus space.

    Parameters:
        knn (int):  Number k of nearest neighbors atoms to consider in the
                    loss computation. Defaults to 4.
    """
    def __init__(self, knn: int = 4):
        """
        Compute MAE loss value between true and predicted edges between atoms,
        taking account of Periodic Boundary Conditions inside 3-D torus space.

        Parameters:
            knn (int):  Number k of nearest neighbors atoms to consider in the
                        loss computation. Defaults to 4.
        """
        super().__init__()

        self.min_distance = MinDistanceLoss()
        self.knn = knn

    def forward(
        self,
        cell: torch.FloatTensor,
        x: torch.FloatTensor,
        x_tilde: torch.FloatTensor,
        num_atoms: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Compute MAE loss value between true and predicted edges between atoms,
        taking account of Periodic Boundary Conditions inside 3-D torus space.

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

        struct_idx = torch.arange(cell.shape[0], device=cell.device)
        batch = struct_idx.repeat_interleave(num_atoms)
        batch_edges = batch[edges.src]

        _, num_edges = torch.unique_consecutive(batch_edges, return_counts=True)

        return self.min_distance(cell, e_tilde_ij, e_ij, num_edges)
