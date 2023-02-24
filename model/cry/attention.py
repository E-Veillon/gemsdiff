import torch
import torch.nn as nn
import torch.nn.functional as F

from ase.data import covalent_radii

from typing import Tuple
import math

from utils.geometry import Geometry
from .gemnet.gemnet import GemNetT

from torch_scatter import scatter_mean, scatter_add

import crystallographic_graph


class Edges(nn.Sequential):
    def __init__(self, features: int, message: int):
        super().__init__(
            nn.Linear(features * 2 + 1, features),
            nn.SiLU(),
            nn.Linear(features, message),
            nn.SiLU(),
        )

    def forward(self, geometry: Geometry, h: torch.FloatTensor) -> torch.FloatTensor:
        inputs = torch.cat(
            (
                h[geometry.edges.src],
                h[geometry.edges.dst],
                geometry.edges_r_ij.unsqueeze(1),
            ),
            dim=1,
        )

        return super().forward(inputs)


class EdgesAlt(nn.Module):
    def __init__(
        self,
        features: int,
        message: int,
        cutoff: float = 10,
        step: float = 0.1,
    ):
        super().__init__()

        self.cutoff = cutoff
        self.step = step
        self.mu = nn.Parameter(
            torch.arange(0, self.cutoff, self.step, dtype=torch.float32),
            requires_grad=False,
        )

        self.gate = nn.Linear(self.mu.data.shape[0], features * 2)
        self.proj = nn.Linear(features * 2, message)

    def forward(self, geometry: Geometry, h: torch.FloatTensor) -> torch.FloatTensor:

        d_ij_emb = torch.exp(
            -1 / self.step * (self.mu[None, :] - geometry.edges_r_ij[:, None]).pow(2)
        )

        gate = self.gate(d_ij_emb)

        h_ij = torch.cat((h[geometry.edges.src], h[geometry.edges.dst]), dim=1)

        return self.proj(gate * h_ij)


class PosUpdate(nn.Sequential):
    def __init__(self, message: int):
        super().__init__(nn.Linear(message, message), nn.SiLU(), nn.Linear(message, 1))

    def forward(self, geometry: Geometry, m_ij: torch.FloatTensor) -> torch.FloatTensor:
        w_ij = super().forward(m_ij)

        x_ij = geometry.x[geometry.edges.src] + geometry.edges_e_ij * w_ij
        x_prime = scatter_mean(
            x_ij, geometry.edges.src, dim=0, dim_size=geometry.x.shape[0]
        )

        return x_prime % 1.0, x_prime - geometry.x


class PosUpdateAtt(nn.Module):
    def __init__(self, message: int):
        super().__init__()

        self.attention = nn.Linear(message, 1)
        self.weigths = nn.Linear(message, 1)

    def forward(self, geometry: Geometry, m_ij: torch.FloatTensor) -> torch.FloatTensor:
        w_ij = self.weigths(m_ij)
        a_ij = self.attention(m_ij).exp()
        a_i = scatter_add(a_ij, geometry.edges.src, dim=0, dim_size=geometry.x.shape[0])
        a_ij = a_ij / a_i[geometry.edges.src]

        x_ij = geometry.edges_e_ij * w_ij * a_ij
        x_diff = scatter_add(
            x_ij, geometry.edges.src, dim=0, dim_size=geometry.x.shape[0]
        )

        return (geometry.x + x_diff) % 1.0, x_diff


"""
class NodeUpdate(nn.Sequential):
    def __init__(self, features: int, message: int):
        super().__init__(
            nn.Linear(features + message, features),
            nn.SiLU(),
            nn.Linear(features, features),
        )

    def forward(
        self, geometry: Geometry, h: torch.FloatTensor, m_ij: torch.FloatTensor
    ) -> torch.FloatTensor:
        m_i = scatter_mean(m_ij, geometry.edges.src, dim=0, dim_size=h.shape[0])
        inputs = torch.cat((h, m_i), dim=1)
        return h + super().forward(inputs)
"""


class NodeUpdate(nn.GRU):
    def __init__(self, features: int, message: int):
        super().__init__(message, features, 1, batch_first=False)

    def forward(
        self, geometry: Geometry, h: torch.FloatTensor, m_ij: torch.FloatTensor
    ) -> torch.FloatTensor:
        m_i = scatter_mean(m_ij, geometry.edges.src, dim=0, dim_size=h.shape[0])
        _, h_prime = super().forward(m_i.unsqueeze(0), h.unsqueeze(0))

        return h_prime.squeeze(0)


class EGNN(nn.Module):
    def __init__(
        self,
        features: int,
        message: int,
        pos_update: bool = True,
        attention: bool = False,
    ):
        super().__init__()

        self.edges = Edges(features, message)
        if pos_update:
            if attention:
                self.pos = PosUpdateAtt(message)
            else:
                self.pos = PosUpdate(message)
        self.nodes = NodeUpdate(features, message)

    def forward(self, geometry: Geometry, h: torch.FloatTensor) -> torch.FloatTensor:
        m_ij = self.edges(geometry, h)

        if hasattr(self, "pos"):
            x_prime, x_diff = self.pos(geometry, m_ij)
        h_prime = self.nodes(geometry, h, m_ij)

        if hasattr(self, "pos"):
            return x_prime, x_diff, h_prime
        else:
            return h_prime


class Denoiser(nn.Module):
    def __init__(
        self,
        features: int
    ):
        super().__init__()

        self.gemnet = GemNetT(features)

    def forward(
        self,
        cell: torch.FloatTensor,
        x_thild: torch.FloatTensor,
        z: torch.FloatTensor,
        num_atoms: torch.LongTensor,
    ) -> torch.FloatTensor:
        
        _, x_traj, _, rho_prime, _ = self.gemnet(cell, x_thild, z, num_atoms)
        return x_traj, rho_prime
