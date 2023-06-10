from typing import Optional, Union

import torch
import torch.nn as nn
from torch_scatter import scatter, scatter_add

from .layers.atom_update_block import OutputBlock
from .layers.base_layers import Dense
from .layers.efficient import EfficientInteractionDownProjection
from .layers.embedding_block import AtomEmbedding, EdgeEmbedding
from .layers.interaction_block import InteractionBlockTripletsOnly
from .layers.radial_basis import RadialBasis
from .layers.spherical_basis import CircularBasisLayer
from .layers.grad.vector_fields import make_vector_fields

from src.utils.geometry import Geometry
from crystallographic_graph import sparse_meshgrid


class GemsNetT(torch.nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_spherical: int = 7,
        num_radial: int = 128,
        num_blocks: int = 3,
        emb_size_atom: int = 128,
        emb_size_edge: int = 128,
        emb_size_trip: int = 32,
        emb_size_rbf: int = 16,
        emb_size_cbf: int = 16,
        emb_size_bil_trip: int = 64,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_concat: int = 1,
        num_atom: int = 3,
        cutoff: float = 6.0,
        rbf: dict = {"name": "gaussian"},
        envelope: dict = {"name": "polynomial", "exponent": 5},
        cbf: dict = {"name": "spherical_harmonics"},
        vector_fields: dict = {
            "type": "grad",
            "edges": [],
            "triplets": ["n_ij", "n_ik", "angle"],
            "normalize": True,
        },
        z_input: Union[str, int] = "embedding",
        energy_targets: int = 1,
        compute_energy: bool = True,
        compute_forces: bool = True,
        compute_stress: bool = True,
        output_init: str = "HeOrthogonal",
        activation: str = "swish",
        scale_file: Optional[str] = None,
    ):
        super().__init__()
        assert num_blocks > 0
        assert z_input == "embedding" or (isinstance(z_input, int) and z_input > 0)

        self.num_blocks = num_blocks

        self.cutoff = cutoff

        ### ---------------------------------- Basis Functions ---------------------------------- ###
        self.radial_basis = RadialBasis(
            num_radial=num_radial,
            cutoff=cutoff,
            rbf=rbf,
            envelope=envelope,
        )

        radial_basis_cbf3 = RadialBasis(
            num_radial=num_radial,
            cutoff=cutoff,
            rbf=rbf,
            envelope=envelope,
        )
        self.cbf_basis3 = CircularBasisLayer(
            num_spherical,
            radial_basis=radial_basis_cbf3,
            cbf=cbf,
            efficient=True,
        )
        ### ------------------------------------------------------------------------------------- ###

        ### ------------------------------- Share Down Projections ------------------------------ ###
        # Share down projection across all interaction blocks
        self.mlp_rbf3 = Dense(
            num_radial,
            emb_size_rbf,
            activation=None,
            bias=False,
        )
        self.mlp_cbf3 = EfficientInteractionDownProjection(
            num_spherical, num_radial, emb_size_cbf
        )

        # Share the dense Layer of the atom embedding block accross the interaction blocks
        self.mlp_rbf_h = Dense(
            num_radial,
            emb_size_rbf,
            activation=None,
            bias=False,
        )
        self.mlp_rbf_out = Dense(
            num_radial,
            emb_size_rbf,
            activation=None,
            bias=False,
        )
        self.mlp_cbf_out = Dense(
            num_spherical,
            emb_size_cbf,
            activation=None,
            bias=False,
        )
        ### ------------------------------------------------------------------------------------- ###

        self.compute_energy = compute_energy
        self.compute_forces = compute_forces
        self.compute_stress = compute_stress

        if self.compute_stress:
            self.vector_fields = make_vector_fields(vector_fields)

        self.output_block = (
            self.compute_energy or self.compute_forces or self.compute_stress
        )

        # Embedding block
        if z_input == "embedding":
            self.atom_emb = AtomEmbedding(emb_size_atom)
        else:
            self.atom_emb = nn.Linear(z_input, emb_size_atom)

        self.atom_latent_emb = nn.Linear(emb_size_atom + latent_dim, emb_size_atom)
        self.edge_emb = EdgeEmbedding(
            emb_size_atom, num_radial, emb_size_edge, activation=activation
        )

        int_blocks = []
        # Interaction Blocks
        interaction_block = InteractionBlockTripletsOnly
        for i in range(num_blocks):
            int_blocks.append(
                interaction_block(
                    emb_size_atom=emb_size_atom,
                    emb_size_edge=emb_size_edge,
                    emb_size_trip=emb_size_trip,
                    emb_size_rbf=emb_size_rbf,
                    emb_size_cbf=emb_size_cbf,
                    emb_size_bil_trip=emb_size_bil_trip,
                    num_before_skip=num_before_skip,
                    num_after_skip=num_after_skip,
                    num_concat=num_concat,
                    num_atom=num_atom,
                    activation=activation,
                    scale_file=scale_file,
                    name=f"IntBlock_{i+1}",
                )
            )

        self.int_blocks = torch.nn.ModuleList(int_blocks)

        num_vector_fields = (
            0 if not hasattr(self, "vector_fields") else self.vector_fields.triplets_dim
        )
        if self.output_block:
            out_blocks = []
            for i in range(num_blocks + 1):
                out_blocks.append(
                    OutputBlock(
                        emb_size_atom=emb_size_atom,
                        emb_size_edge=emb_size_edge,
                        emb_size_trip=emb_size_trip,
                        emb_size_rbf=emb_size_rbf,
                        emb_size_cbf=emb_size_cbf,
                        nHidden=num_atom,
                        num_targets=energy_targets,
                        num_vector_fields=num_vector_fields,
                        activation=activation,
                        output_init=output_init,
                        direct_forces=True,
                        scale_file=scale_file,
                        name=f"OutBlock_{i}",
                    )
                )

            self.out_blocks = torch.nn.ModuleList(out_blocks)

        self.shared_parameters = [
            (self.mlp_rbf3, self.num_blocks),
            (self.mlp_cbf3, self.num_blocks),
            (self.mlp_rbf_h, self.num_blocks),
            (self.mlp_rbf_out, self.num_blocks + 1),
        ]

    def forward(
        self,
        z: torch.LongTensor,
        geometry: Geometry,
        emb: torch.FloatTensor = None,
    ):
        cell = geometry.cell
        x = geometry.x
        num_atoms = geometry.num_atoms
        batch = geometry.batch
        idx_s = geometry.edges.src
        idx_t = geometry.edges.dst
        D_st = geometry.edges_r_ij
        V_st = -geometry.edges_v_ij / geometry.edges_r_ij[:, None]
        U_st = -geometry.edges_e_ij / geometry.edges_r_ij[:, None]
        id_swap = geometry.edges.reverse_idx

        num_edges = scatter_add(
            torch.ones_like(idx_s), idx_s, dim=0, dim_size=x.shape[0]
        )
        i_triplets, j_triplets = sparse_meshgrid(num_edges)

        mask = i_triplets != j_triplets
        id3_ba = i_triplets[mask]
        id3_ca = j_triplets[mask]

        # Calculate triplet angles
        cosφ_cab = torch.sum(V_st[id3_ca] * V_st[id3_ba], dim=-1).clamp(min=-1, max=1)
        rad_cbf3, sbf3 = self.cbf_basis3(D_st, cosφ_cab, id3_ca)

        rbf = self.radial_basis(D_st)

        # Embedding block
        if emb is None:
            h = self.atom_emb(z)
        else:
            h = self.atom_emb(z) + emb[geometry.batch]

        m = self.edge_emb(h, rbf, idx_s, idx_t)

        rbf3 = self.mlp_rbf3(rbf)
        cbf3 = self.mlp_cbf3(rad_cbf3, sbf3)

        rbf_h = self.mlp_rbf_h(rbf)
        rbf_out = self.mlp_rbf_out(rbf)
        cbf_out = self.mlp_cbf_out(sbf3)

        if self.output_block:
            E_t, F_st, S_st = self.out_blocks[0](
                h, m, rbf_out, cbf_out, idx_t, id3_ba, id3_ca
            )

        for i in range(self.num_blocks):
            h, m = self.int_blocks[i](
                h=h,
                m=m,
                rbf3=rbf3,
                cbf3=cbf3,
                id_swap=id_swap,
                id3_ba=id3_ba,
                id3_ca=id3_ca,
                rbf_h=rbf_h,
                idx_s=idx_s,
                idx_t=idx_t,
            )

            if self.output_block:
                E, F, S = self.out_blocks[i + 1](
                    h, m, rbf_out, cbf_out, idx_t, id3_ba, id3_ca
                )
                E_t += E
                F_st += F
                S_st += S

        results = [h]

        if self.compute_energy:
            # ========================== ENERGY ==========================
            E_t = scatter(E_t, batch, dim=0, dim_size=num_atoms.shape[0], reduce="mean")

            results.append(E_t)

        if self.compute_forces:
            # ========================== FORCES ==========================
            F_st_vec = F_st[:, :, None] * U_st[:, None, :]

            F_t = scatter(
                F_st_vec,
                idx_t,
                dim=0,
                dim_size=num_atoms.sum(),
                reduce="add",
            )
            F_t = F_t.squeeze(1)

            results.append((x + F_t) % 1.0)
            results.append(F_t)

        if self.compute_stress:
            # ========================== STRESS ==========================
            batch_triplets = batch[idx_s[id3_ba]]

            e_ij = geometry.edges_e_ij[id3_ba]
            e_ik = geometry.edges_e_ij[id3_ca]

            vector_fields = self.vector_fields(cell, batch_triplets, e_ij, e_ik)

            filter_nan = ~(
                (vector_fields != vector_fields)
                .view(vector_fields.shape[0], -1)
                .any(dim=1)
            )
            batch_triplets = batch_triplets[filter_nan]
            fields = (S_st[filter_nan, :, None, None] * vector_fields[filter_nan]).sum(
                dim=1
            )
            I = torch.eye(3, 3, device=cell.device)[None]
            S_t = I + scatter(
                fields,
                batch_triplets,
                dim=0,
                dim_size=cell.shape[0],
                reduce="mean",
            )  # 1st order approx of matrix exp

            results.append(S_t)

        return tuple(results)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
