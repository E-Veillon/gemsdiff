import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

from src.utils.geometry import Geometry
from .gemsnet import GemsNetT


class GemsNetDenoiser(nn.Module):
    def __init__(
        self,
        features: int,
        knn: int,
        num_blocks: int,
        vector_fields: dict,
    ):
        super().__init__()

        self.knn = knn

        self.gemsnet = GemsNetT(
            features,
            vector_fields=vector_fields,
            compute_energy=False,
            compute_forces=True,
            compute_stress=True,
        )

    def forward(
        self,
        cell: torch.FloatTensor,
        x: torch.FloatTensor,
        z: torch.FloatTensor,
        num_atoms: torch.LongTensor,
    ) -> torch.FloatTensor:
        geometry = Geometry(
            cell,
            num_atoms,
            x,
            knn=self.knn,
            triplets=False,
            symetric=True,
            compute_reverse_idx=True,
        )

        _, x_prime, x_traj, rho_prime = self.gemsnet(z, geometry)
        return x_prime, x_traj, rho_prime
