import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

from ase.data import covalent_radii

from typing import Tuple
import math

from utils.geometry import Geometry
from utils.shape import build_shapes, assert_tensor_match, shape
from model.cry.gnn import MPNN, Edges

import crystallographic_graph


class Adj(nn.Module):
    def __init__(self, layers: int, features: int, knn: int = 32, z_max: int = 100):
        super().__init__()

        self.knn = knn

        self.embedding = nn.Embedding(z_max, features)
        self.mpnn = nn.ModuleList([MPNN(features) for _ in range(layers)])
        self.adj = Edges(features, features, 1)

        self.covalent_radii = nn.Parameter(
            torch.from_numpy(covalent_radii).float(), requires_grad=False
        )
        idx = torch.tensor([-1, 0, 1])
        self.offset = nn.Parameter(
            torch.stack(torch.meshgrid(idx, idx, idx), dim=-1).view(-1, 3),
            requires_grad=False,
        )

    def get_full_adj(self, num_atoms: torch.LongTensor) -> crystallographic_graph.Edges:
        i, j = crystallographic_graph.sparse_meshgrid(num_atoms)

        print(i.shape, j.shape, self.offset.shape)
        print(i.unsqueeze(1).unsqueeze(2).repeat(1, 27, 1).shape)

        edges = torch.cat(
            (
                i.unsqueeze(1).unsqueeze(2).repeat(1, 27, 1),
                j.unsqueeze(1).unsqueeze(2).repeat(1, 27, 1),
                self.offset.unsqueeze(0).repeat(i.shape[0], 1, 1),
            ),
            dim=2,
        ).view(-1, 5)
        print(edges.shape)

        exit(0)
        struct_idx = torch.arange(num_atoms.shape[0], device=num_atoms.device)
        batch = struct_idx.repeat_interleave(num_atoms)

        max_radii = self.covalent_radii[z].max()
        edges = crystallographic_graph.make_graph(
            cell, x, num_atoms, cutoff=max_radii * 2 * margin
        )

        d_ij = torch.bmm(
            cell[batch[edges.src]],
            (x[edges.src] + edges.cell.float() - x[edges.dst]).unsqueeze(2),
        ).norm(dim=(1, 2))

        threshold = (
            self.covalent_radii[z[edges.src]] + self.covalent_radii[z[edges.dst]]
        ) * margin
        mask = d_ij < threshold

        edges.src = edges.src[mask]
        edges.dst = edges.dst[mask]
        edges.cell = edges.cell[mask]

        return edges

    def get_graph(
        self,
        cell: torch.FloatTensor,
        x: torch.FloatTensor,
        z: torch.LongTensor,
        num_atoms: torch.LongTensor,
        margin: float = 1.2,
    ) -> crystallographic_graph.Edges:
        struct_idx = torch.arange(num_atoms.shape[0], device=num_atoms.device)
        batch = struct_idx.repeat_interleave(num_atoms)

        max_radii = self.covalent_radii[z].max()
        edges = crystallographic_graph.make_graph(
            cell, x, num_atoms, cutoff=max_radii * 2 * margin
        )

        d_ij = torch.bmm(
            cell[batch[edges.src]],
            (x[edges.src] + edges.cell.float() - x[edges.dst]).unsqueeze(2),
        ).norm(dim=(1, 2))

        threshold = (
            self.covalent_radii[z[edges.src]] + self.covalent_radii[z[edges.dst]]
        ) * margin
        mask = d_ij < threshold

        edges.src = edges.src[mask]
        edges.dst = edges.dst[mask]
        edges.cell = edges.cell[mask]

        return edges

    def forward(
        self,
        cell: torch.FloatTensor,
        x: torch.FloatTensor,
        x_thild: torch.FloatTensor,
        z: torch.LongTensor,
        num_atoms: torch.LongTensor,
    ) -> torch.FloatTensor:
        self.get_full_adj(num_atoms)

        geometry = Geometry(cell, num_atoms, x_thild, knn=self.knn)

        geometry.x = x
        geometry.update_vectors()
        threshold = (
            self.covalent_radii[z[geometry.edges.src]]
            + self.covalent_radii[z[geometry.edges.dst]]
        ) * 1.2
        geometry.filter_edges(geometry.edges_r_ij < threshold)

        edges = self.get_graph(cell, x, z, num_atoms)

        print(geometry.edges.src.shape, edges.src.shape)
