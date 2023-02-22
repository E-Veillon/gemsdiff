import torch
import torch.nn as nn

from .gemnet import GemNetT


class GemNet(nn.Module):
    def __init__(self, hidden_dim=128, latent_dim=256, max_neighbors=20, radius=6.0):
        super().__init__()
        self.cutoff = radius
        self.max_num_neighbors = max_neighbors

        self.gemnet = GemNetT(
            num_targets=1,
            latent_dim=latent_dim,
            emb_size_atom=hidden_dim,
            emb_size_edge=hidden_dim,
            regress_forces=True,
            cutoff=self.cutoff,
            max_neighbors=self.max_num_neighbors,
            otf_graph=True,
        )

    def forward(
        self,
        cell: torch.FloatTensor,
        x: torch.FloatTensor,
        x_thild: torch.FloatTensor,
        z: torch.FloatTensor,
        num_atoms: torch.LongTensor,
    )-> torch.FloatTensor:
        #z, frac_coords, atom_types, num_atoms, lengths, angles
