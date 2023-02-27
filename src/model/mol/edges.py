import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_mean

import crystallographic_graph


class Edges(nn.Sequential):
    def __init__(self, features: int, message: int):
        super().__init__(
            nn.Linear(features * 3 + 1, features),
            nn.SiLU(),
            nn.Linear(features, message),
            nn.SiLU(),
        )

    def forward(
        self,
        x: torch.FloatTensor,
        h: torch.FloatTensor,
        h_ij: torch.FloatTensor,
        edges: torch.LongTensor,
    ) -> torch.FloatTensor:
        d_ij = (x[edges[0]] - x[edges[1]]).norm(dim=1).unsqueeze(1)
        inputs = torch.cat((h[edges[0]], h[edges[1]], h_ij, d_ij), dim=1)

        return super().forward(inputs)


class PosUpdate(nn.Sequential):
    def __init__(self, message: int):
        super().__init__(nn.Linear(message, message), nn.SiLU(), nn.Linear(message, 1))

    def forward(
        self, x: torch.FloatTensor, m_ij: torch.FloatTensor, edges: torch.LongTensor
    ) -> torch.FloatTensor:
        w_ij = super().forward(m_ij)

        x_ij = (x[edges[1]] - x[edges[0]]) * w_ij
        x_diff = scatter_mean(x_ij, edges[0], dim=0, dim_size=x.shape[0])

        return x + x_diff


class EdgeUpdate(nn.Sequential):
    def __init__(self, features: int):
        super().__init__(
            nn.Linear(features * 3, features), nn.SiLU(), nn.Linear(features, features)
        )

    def forward(
        self,
        h: torch.FloatTensor,
        edges: torch.LongTensor,
        h_ij: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        if h_ij is None:
            h_ij = torch.zeros(
                (edges.shape[1], h.shape[1]), dtype=torch.float32, device=h.device
            )
        inputs = torch.cat((h[edges[0]], h[edges[1]], h_ij), dim=1)
        return h_ij + super().forward(inputs)


class NodeUpdate(nn.Sequential):
    def __init__(self, features: int, message: int):
        super().__init__(
            nn.Linear(features + message, features),
            nn.SiLU(),
            nn.Linear(features, features),
        )

    def forward(
        self, h: torch.FloatTensor, m_ij: torch.FloatTensor, edges: torch.LongTensor
    ) -> torch.FloatTensor:
        m_i = scatter_mean(m_ij, edges[0], dim=0, dim_size=h.shape[0])
        inputs = torch.cat((h, m_i), dim=1)
        return h + super().forward(inputs)


class EGNN(nn.Module):
    def __init__(self, features: int, message: int):
        super().__init__()

        self.messages = Edges(features, message)
        self.pos = PosUpdate(message)
        self.edges = EdgeUpdate(features)
        self.nodes = NodeUpdate(features, message)

    def forward(
        self,
        x: torch.FloatTensor,
        h: torch.FloatTensor,
        edges: torch.LongTensor,
        h_ij: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        h_prime_ij = self.edges(h, edges, h_ij=h_ij)
        m_ij = self.messages(x, h, h_prime_ij, edges)
        x_prime = self.pos(x, m_ij, edges)
        h_prime = self.nodes(h, m_ij, edges)

        return x_prime, h_prime, h_prime_ij


class Denoise(nn.Module):
    def __init__(
        self,
        features: int,
        layers: int,
        message: int,
        z_max: int = 100,
        knn: int = 16,
        gen_graph: bool = True,
    ):
        super().__init__()

        self.gen_graph = gen_graph
        self.knn = knn

        self.embedding = nn.Embedding(z_max, features)
        self.egnn = nn.ModuleList([EGNN(features, message) for _ in range(layers)])

    @torch.no_grad()
    def get_edges(
        self, x: torch.FloatTensor, num_atoms: torch.LongTensor
    ) -> torch.LongTensor:
        i, j = crystallographic_graph.sparse_meshgrid(num_atoms)
        mask = i != j
        i, j = i[mask], j[mask]

        d_ij = (x[i] - x[j]).norm(dim=1)
        d_ij, idx = d_ij.sort()
        i, j = i[idx], j[idx]

        i, idx = i.sort(stable=True)
        d_ij = d_ij[idx]
        j = j[idx]

        _, counts = torch.unique_consecutive(i, return_counts=True)
        idx = F.pad(counts, (1, 0)).cumsum(0)

        k_th = torch.minimum(idx[:-1] + self.knn, idx[1:] - 1)
        threshold = d_ij[k_th].repeat_interleave(counts)

        mask = d_ij < threshold
        edges = torch.stack((i[mask], j[mask]), dim=0)

        return edges

    def forward(
        self,
        x: torch.FloatTensor,
        z: torch.LongTensor,
        num_atoms: torch.LongTensor,
        edges: torch.LongTensor,
    ) -> torch.FloatTensor:
        edges_alt = self.get_edges(x, num_atoms)

        n = x.shape[0]
        mask = edges_alt[0] < edges_alt[1]
        i = torch.where(mask, edges_alt[0], edges_alt[1])
        j = torch.where(mask, edges_alt[1], edges_alt[0])
        idx = torch.unique(i * n + j)
        i = torch.div(idx, n, rounding_mode="floor")
        j = idx % n
        edges_alt = torch.stack(
            (torch.cat((i, j), dim=0), torch.cat((j, i), dim=0)), dim=0
        )

        if self.gen_graph:
            selected_edges = edges_alt
        else:
            selected_edges = edges

        h = self.embedding(z)

        h_ij = None

        for layer in self.egnn:
            x, h, h_ij = layer(x, h, selected_edges, h_ij)

        return x
