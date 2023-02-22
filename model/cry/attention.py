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
        features: int,
        mpnn_layers: int,
        egnn_layers: int,
        hidden_dim: int = 64,
        knn: int = 32,
        z_max: int = 100,
        hard_attention: bool = True,
    ):
        super().__init__()

        self.knn = knn

        """
        self.embedding = nn.Embedding(z_max, features)
        self.mpnn = nn.ModuleList(
            [EGNN(features, hidden_dim, pos_update=False) for _ in range(mpnn_layers)]
        )
        self.actions = nn.ModuleList([EGNN(features, hidden_dim, pos_update=True, attention=False) for _ in range(egnn_layers)])
        # self.actions = EGNN(features, hidden_dim, pos_update=True, attention=True)
        """

        self.gemnet = GemNetT(features)

        self.covalent_radii = nn.Parameter(
            torch.from_numpy(covalent_radii).float(), requires_grad=False
        )

        self.attention = Edges(features, 1)

        idx = torch.tensor([-1, 0, 1])
        self.offset = nn.Parameter(
            torch.stack(torch.meshgrid(idx, idx, idx), dim=-1).view(-1, 3),
            requires_grad=False,
        )

        self.hard_attention = hard_attention

    def mask_covalent(
        self,
        cell: torch.FloatTensor,
        x: torch.FloatTensor,
        z: torch.FloatTensor,
        batch: torch.LongTensor,
        edges: crystallographic_graph.Edges,
        margin: float = 2.0,
    ) -> torch.BoolTensor:
        e_ij = x[edges.dst, :] + edges.cell - x[edges.src, :]
        r_ij = torch.bmm(cell[batch[edges.src]], e_ij.unsqueeze(2)).norm(dim=(1, 2))

        threshold = (
            self.covalent_radii[z[edges.src]] + self.covalent_radii[z[edges.dst]]
        ) * margin

        mask = r_ij < threshold

        return mask

    @torch.no_grad()
    def get_graph(
        self, cell: torch.FloatTensor, x: torch.FloatTensor, num_atoms: torch.LongTensor
    ) -> Geometry:
        j, i = crystallographic_graph.sparse_meshgrid(num_atoms)
        mask = j != i
        i, j = i[mask], j[mask]

        edges = crystallographic_graph.Edges(
            src=i.repeat_interleave(self.offset.shape[0]),
            dst=j.repeat_interleave(self.offset.shape[0]),
            cell=self.offset.repeat(i.shape[0], 1),
        )

        geometry = Geometry(cell, num_atoms, x, triplets=False, edges_idx=edges)

        return geometry

    def is_in(
        self, edges: crystallographic_graph.Edges, others: crystallographic_graph.Edges
    ) -> torch.BoolTensor:
        edges = torch.cat(
            (edges.src.unsqueeze(1), edges.dst.unsqueeze(1), edges.cell), dim=1
        )
        others = torch.cat(
            (others.src.unsqueeze(1), others.dst.unsqueeze(1), others.cell), dim=1
        )

        min_edges = torch.minimum(edges.min(dim=0).values, others.min(dim=0).values)
        span_edges = (
            torch.maximum(edges.max(dim=0).values, others.max(dim=0).values) - min_edges
        )

        cumprod = F.pad(span_edges, (1, 0), value=1.0).cumprod(0)
        offset = cumprod[:-1]

        edges_idx = ((edges - min_edges) * offset).sum(dim=1)
        others_idx = ((others - min_edges) * offset).sum(dim=1)

        others_idx = others_idx.sort().values
        find_idx = torch.bucketize(edges_idx, others_idx)
        mask = edges_idx == others_idx[find_idx.clamp(0, others_idx.shape[0] - 1)]

        return mask

    def forward(
        self,
        cell: torch.FloatTensor,
        x: torch.FloatTensor,
        x_thild: torch.FloatTensor,
        z: torch.FloatTensor,
        num_atoms: torch.LongTensor,
    ) -> torch.FloatTensor:
        # geometry = self.get_graph(cell, x_thild, num_atoms)
        # close_geometry = Geometry(cell, num_atoms, x, knn=16, triplets=False)

        # mask = self.is_in(geometry.edges, close_geometry.edges).unsqueeze(1)
        # geometry.filter_edges(mask)

        x_prime, _, _ = self.gemnet(cell, x_thild, z, num_atoms)
        return x_prime

        geometry = Geometry(cell, num_atoms, x_thild, knn=self.knn, triplets=False)

        h = self.embedding(z)

        for layer in self.mpnn:
            h = layer(geometry, h)

        for layer in self.actions:
            x_prime, _, h = self.actions[0](geometry, h)
            # x_prime, x_diff, h = layer(geometry, h)
            geometry.x = x_prime
            geometry.update_vectors()

        return x_prime
