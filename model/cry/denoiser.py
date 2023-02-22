import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
import math

from utils.geometry import Geometry
from utils.shape import build_shapes, assert_tensor_match, shape
from .gnn import MPNN, Actions


class Denoiser(nn.Module):
    def __init__(
        self,
        features: int,
        mpnn_layers: int,
        hidden_dim: int = 64,
        z_max: int = 100,
        knn: int = 16,
    ):
        super().__init__()

        self.knn = knn

        self.embedding = nn.Embedding(z_max, features)
        self.mpnn = nn.ModuleList([MPNN(features) for _ in range(mpnn_layers)])
        self.actions = Actions(features, hidden_dim)

    def forward(
        self,
        cell: torch.FloatTensor,
        x: torch.FloatTensor,
        z: torch.FloatTensor,
        struct_size: torch.LongTensor,
    ) -> torch.FloatTensor:
        geometry = Geometry(cell, struct_size, x % 1, knn=self.knn)

        h = self.embedding(z)

        for layer in self.mpnn:
            h = layer(geometry, h)

        x_prime = self.actions(geometry, h)

        return x_prime
