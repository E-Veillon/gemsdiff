import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_mean

import crystallographic_graph


class Edges(nn.Sequential):
    def __init__(self, features: int, message: int):
        super().__init__(
            nn.Linear(features * 2 + 1, features),
            nn.SiLU(),
            nn.Linear(features, message),
            nn.SiLU(),
        )

    def forward(
        self, x: torch.FloatTensor, h: torch.FloatTensor, edges: torch.LongTensor
    ) -> torch.FloatTensor:
        d_ij = (x[edges[0]] - x[edges[1]]).norm(dim=1).unsqueeze(1)
        inputs = torch.cat((h[edges[0]], h[edges[1]], d_ij), dim=1)

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


class PosUpdateAlt(nn.Sequential):
    def __init__(self, message: int):
        super().__init__(nn.Linear(message, message), nn.SiLU(), nn.Linear(message, 1))

    def forward(
        self, x: torch.FloatTensor, m_ij: torch.FloatTensor, edges: torch.LongTensor
    ) -> torch.FloatTensor:
        w_ij = super().forward(m_ij)

        x_ij = F.normalize((x[edges[1]] - x[edges[0]]), dim=1) * w_ij
        x_diff = scatter_mean(x_ij, edges[0], dim=0, dim_size=x.shape[0])

        return x + x_diff


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

        self.edges = Edges(features, message)
        self.pos = PosUpdate(message)
        self.nodes = NodeUpdate(features, message)

    def forward(
        self, x: torch.FloatTensor, h: torch.FloatTensor, edges: torch.LongTensor
    ) -> torch.FloatTensor:
        m_ij = self.edges(x, h, edges)
        x_prime = self.pos(x, m_ij, edges)
        h_prime = self.nodes(h, m_ij, edges)

        return x_prime, h_prime


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

        self.knn = knn
        self.gen_graph = gen_graph

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

    def backup(self, filename: str, edges, edges_alt, num_atoms):
        mask = edges_alt[0] < num_atoms
        filtered_alt = edges_alt[:, mask]

        mask = edges[0] < num_atoms
        filtered = edges[:, mask]

        from torchvision.utils import save_image

        img = torch.zeros((num_atoms * 2, num_atoms), dtype=torch.float32)
        img[filtered[0], filtered[1]] = 1
        img[filtered_alt[0] + num_atoms, filtered_alt[1]] = 1
        save_image(img, filename)

        exit(0)

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

        for layer in self.egnn:
            x, h = layer(x, h, selected_edges)

        return x
