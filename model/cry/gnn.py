import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter, scatter_add

from typing import Tuple
import math

from utils.geometry import Geometry
from utils.shape import build_shapes, assert_tensor_match, shape


class Edges(nn.Module):
    def __init__(
        self,
        features: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 0,
        cutoff: float = 10,
        step: float = 0.1,
        bias: bool = False,
    ):
        super(Edges, self).__init__()

        if output_dim is None:
            output_dim = features

        self.cutoff = cutoff
        self.step = step
        self.mu = nn.Parameter(
            torch.arange(0, self.cutoff, self.step, dtype=torch.float32)
        )

        layers = [
            nn.Linear(2 * features + self.mu.shape[0], hidden_dim, bias=False),
            nn.SiLU(),
        ]
        for _ in range(n_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim, bias=False), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, output_dim, bias=bias))
        self.mlp = nn.Sequential(*layers)

        self.reset_parameters()

    def reset_parameters(self, gain=1.0):
        count = sum([isinstance(layer, nn.Linear) for layer in self.mlp])

        layer_gain = math.exp(math.log(gain) / count)

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight, gain=layer_gain)

    def forward(self, h, src, dst, edge_norm):
        d_ij_emb = torch.exp(
            -1 / self.step * (self.mu[None, :] - edge_norm[:, None]).pow(2)
        )
        inputs = torch.cat((h[src], h[dst], d_ij_emb), dim=-1)

        return self.mlp(inputs)


class UpdateFeatures(nn.GRU):
    def __init__(self, features: int):
        super(UpdateFeatures, self).__init__(features, features, 1, batch_first=False)

    def forward(self, h: torch.FloatTensor, mi: torch.FloatTensor):
        _, h_prime = super().forward(mi.unsqueeze(0), h.unsqueeze(0))

        return h_prime.squeeze(0)


class Actions(nn.Module):
    def __init__(
        self,
        features: int,
        hidden_dim: int,
        attention: bool = False,
        reduce_pos: str = "mean",
    ):
        super(Actions, self).__init__()

        self.reduce_pos = reduce_pos

        if attention:
            self.attention = Edges(
                features,
                output_dim=1,
                hidden_dim=hidden_dim,
                n_layers=0,
                bias=False,
            )

        self.edges = Edges(
            features,
            output_dim=1,
            hidden_dim=hidden_dim,
            n_layers=1,
            bias=True,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.edges.reset_parameters()

        if hasattr(self, "attention"):
            self.attention.reset_parameters()

    def forward(self, geometry: Geometry, h: torch.FloatTensor) -> torch.FloatTensor:
        weights = self.edges(
            h, geometry.edges.src, geometry.edges.dst, geometry.edges_r_ij
        )
        print("weights", (weights != weights).any().item())

        if hasattr(self, "attention"):
            a_ij = self.attention(
                h, geometry.edges.src, geometry.edges.dst, geometry.edges_r_ij
            ).exp()

            print("a_ij", (a_ij != a_ij).any().item(), (a_ij == 0).any().item())

            # print("a_ij", (a_ij != a_ij).any())
            # print("a_ij", a_ij)

            a_i = scatter_add(a_ij, geometry.edges.src, dim=0, dim_size=h.shape[0])
            print("a_i", (a_i != a_i).any().item(), (a_i == 0).any().item())
            # print("a_i", (a_i != a_i).any(), (a_i == 0).any())
            a_ij = a_ij / (a_i[geometry.edges.src] + 1e-6)
            weights = weights * a_ij

        x_diff = scatter(
            weights * geometry.edges_e_ij,
            geometry.edges.src,
            dim=0,
            dim_size=geometry.x.shape[0],
            reduce=self.reduce_pos,
        )
        print("x_diff", (x_diff != x_diff).any().item())

        return (geometry.x + x_diff) % 1.0


class MPNN(nn.Module):
    def __init__(self, features: int):
        super(MPNN, self).__init__()

        self.message_f = Edges(
            features, hidden_dim=features, output_dim=features, n_layers=0
        )
        self.update_f = UpdateFeatures(features)

        self.reset_parameters()

    def reset_parameters(self):
        self.message_f.reset_parameters()
        self.update_f.reset_parameters()

    def forward(self, geometry: Geometry, h: torch.FloatTensor):
        # message passing
        mij = self.message_f(
            h, geometry.edges.src, geometry.edges.dst, geometry.edges_r_ij
        )
        mi = scatter(mij, geometry.edges.src, dim=0, reduce="mean", dim_size=h.shape[0])
        h_prime = self.update_f(h, mi)

        return h_prime
