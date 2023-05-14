import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
import os

from src.utils.geometry import Geometry
from .gemsnet import GemsNetT


class GemsNetVAE(nn.Module):
    def __init__(
        self,
        features: int,
        knn: int,
        num_blocks: int,
        vector_fields: dict,
        global_features: int = None,
        emb_size_atom: int = 128,
    ):
        super().__init__()

        if global_features is None:
            global_features = features

        self.knn = knn

        # energy_targets
        self.encoder = GemsNetT(
            features,
            num_blocks=num_blocks,
            emb_size_atom=emb_size_atom,
            energy_targets=global_features,
            compute_energy=True,
            compute_forces=False,
            compute_stress=False,
        )
        self.decoder = GemsNetT(
            features,
            num_blocks=num_blocks,
            z_input=emb_size_atom + global_features,
            compute_energy=False,
            compute_forces=True,
            compute_stress=True,
            vector_fields=vector_fields,
        )

    def forward(
        self,
        cell: torch.FloatTensor,
        x: torch.FloatTensor,
        z: torch.LongTensor,
        num_atoms: torch.LongTensor,
        emb: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        eye = torch.eye(3, device=cell.device).unsqueeze(0).repeat(cell.shape[0], 1, 1)

        geometry = Geometry(
            cell,
            num_atoms,
            x,
            knn=self.knn,
            triplets=False,
            symetric=True,
            compute_reverse_idx=True,
        )

        h, h_mat = self.encoder(z, geometry, emb)

        # print("h:", h.mean().item(), h.std().item())
        # print("h_mat:", h_mat.mean().item(), h.std().item())

        h_atoms = torch.cat((h, h_mat[geometry.batch]), dim=1)

        geometry = Geometry(
            eye,
            num_atoms,
            x,
            knn=self.knn,
            triplets=False,
            symetric=True,
            compute_reverse_idx=True,
        )

        _, x_prime, x_traj, rho_prime = self.decoder(h_atoms, geometry)

        return x_prime, x_traj, rho_prime

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
