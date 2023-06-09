import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_add
import tqdm

from typing import Tuple, Union
import math
import warnings

from src.utils.scaler import LatticeScaler
from src.utils.geometry import Geometry
from src.loss import OptimalTrajLoss, LatticeParametersLoss
from .vae import GemsNetVAE
from .gemsnet import GemsNetT

# import scipy.linalg

from ase.data import atomic_masses


"""
def logm(X):
    return torch.stack(
        [
            torch.from_numpy(scipy.linalg.logm(X_i.cpu(), disp=False)[0]).real.float()
            for X_i in X
        ],
        0,
    ).to(X.device)
"""


def logm(X: torch.FloatTensor) -> torch.FloatTensor:
    w, V = torch.linalg.eig(X)
    return torch.einsum("bij,bj,bjk->bik", V, w.log(), torch.linalg.inv(V)).real


class TEmbedding(nn.Module):
    def __init__(self, dim: int, T: int):
        assert (dim % 2) == 0

        super().__init__()

        step = torch.arange(0, dim // 2).float()
        self.step = nn.Parameter(torch.pow(T, step * 2 / dim), requires_grad=False)

    def forward(self, t: torch.LongTensor) -> torch.FloatTensor:
        x = t[:, None] / self.step[None, :]
        emb = torch.cat((x.cos(), x.sin()), dim=1)

        return emb


class TrajLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, pred: torch.FloatTensor, target: torch.FloatTensor
    ) -> torch.FloatTensor:
        return (pred - target).norm(dim=1).pow(2).mean()


class GemsNetDiffusion(nn.Module):
    def __init__(
        self,
        lattice_scaler: LatticeScaler,
        features: int = 256,
        knn: int = 32,
        num_blocks: int = 3,
        vector_fields: dict = {
            "type": "grad",
            "normalize": True,
            "edges": [],
            "triplets": ["n_ij", "n_ik", "angle"],
        },
        global_features: int = None,
        emb_size_atom: int = 128,
        x_betas: Tuple[float, float] = (1e-6, 2e-4),
        rho_betas: Tuple[float, float] = (1e-5, 1e-1),
        diffusion_steps: int = 100,
        limit_density: Tuple[float, float] = (0.1, 100.0),
    ):
        super().__init__()

        self.t_embedding = TEmbedding(emb_size_atom, diffusion_steps)

        # energy_targets
        self.knn = knn
        self.gnn = GemsNetT(
            features,
            num_blocks=num_blocks,
            energy_targets=1,
            compute_energy=False,
            compute_forces=True,
            compute_stress=True,
            vector_fields=vector_fields,
        )
        """
        self.decoder = GemsNetT(
            features,
            num_blocks=num_blocks,
            z_input=emb_size_atom + global_features,
            compute_energy=False,
            compute_forces=True,
            compute_stress=True,
        )

        self.gemsnet = GemsNetVAE(
            features=features,
            knn=knn,
            num_blocks=num_blocks,
            vector_fields=vector_fields,
            global_features=global_features,
            emb_size_atom=emb_size_atom,
        )
        """
        self.loss_lattice_fn = LatticeParametersLoss(lattice_scaler=lattice_scaler)
        # self.loss_pos_fn = TrajLoss()
        self.loss_pos_fn = OptimalTrajLoss(center=True, euclidian=True, distance="l1")

        (self.density_min, self.density_max) = limit_density
        self.x_betas = nn.Parameter(
            torch.linspace(x_betas[0], x_betas[1], diffusion_steps),
            requires_grad=False,
        )
        self.x_sigma = nn.Parameter(self.x_betas.sqrt(), requires_grad=False)
        self.x_alphas = nn.Parameter(1 - self.x_betas, requires_grad=False)
        self.x_alphas_bar = nn.Parameter(
            torch.cumprod(self.x_alphas, dim=0), requires_grad=False
        )
        self.rho_betas = nn.Parameter(
            torch.linspace(rho_betas[0], rho_betas[1], diffusion_steps),
            requires_grad=False,
        )
        self.rho_sigma = nn.Parameter(self.rho_betas.sqrt(), requires_grad=False)
        self.rho_alphas = nn.Parameter(1 - self.rho_betas, requires_grad=False)
        self.rho_alphas_bar = nn.Parameter(
            torch.cumprod(self.rho_alphas, dim=0), requires_grad=False
        )
        self.basis = nn.Parameter(
            torch.tensor(
                [
                    [
                        [0.0, 1 / math.sqrt(2), 0.0],
                        [-1 / math.sqrt(2), 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 1 / math.sqrt(2)],
                        [0.0, 0.0, 0.0],
                        [-1 / math.sqrt(2), 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 1 / math.sqrt(2)],
                        [0.0, -1 / math.sqrt(2), 0.0],
                    ],
                    [
                        [0.0, 1 / math.sqrt(2), 0.0],
                        [1 / math.sqrt(2), 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 1 / math.sqrt(2)],
                        [0.0, 0.0, 0.0],
                        [1 / math.sqrt(2), 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 1 / math.sqrt(2)],
                        [0.0, 1 / math.sqrt(2), 0.0],
                    ],
                    [
                        [1 / math.sqrt(2), 0.0, 0.0],
                        [0.0, -1 / math.sqrt(2), 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                    [
                        [1 / math.sqrt(6), 0.0, 0.0],
                        [0.0, 1 / math.sqrt(6), 0.0],
                        [0.0, 0.0, -2 / math.sqrt(6)],
                    ],
                    [
                        [1 / math.sqrt(3), 0.0, 0.0],
                        [0.0, 1 / math.sqrt(3), 0.0],
                        [0.0, 0.0, 1 / math.sqrt(3)],
                    ],
                ]
            ),
            requires_grad=False,
        )
        self.atomic_masses = nn.Parameter(
            torch.from_numpy(atomic_masses).float(), requires_grad=False
        )

        self.basis_inv = nn.Parameter(
            torch.inverse(self.basis.view(9, 9)).view(3, 3, 9),
            requires_grad=False,
        )

    @property
    def diffusion_steps(self) -> int:
        return self.x_betas.shape[0]

    def random_rho_T(self, n: int, device: torch.device = None) -> torch.FloatTensor:
        epsilon = torch.randn((n, 6), device=device)
        epsilon = F.pad(epsilon, (3, 0), value=0.0)
        epsilon[:, -1] *= 0.25

        return self.vect_to_rho(epsilon)

    def rho_to_vect(self, rho: torch.FloatTensor) -> torch.FloatTensor:
        return torch.einsum("bik,ikl->bl", logm(rho), self.basis_inv)

    def vect_to_rho(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return torch.matrix_exp(torch.einsum("bi,ijk->bjk", x, self.basis))

    def get_mu_rho_t(
        self,
        x_t: torch.FloatTensor,
        x_0: torch.FloatTensor,
        t: int,
    ) -> torch.FloatTensor:
        c_0 = ((self.rho_alphas_bar[t - 1]).sqrt() * self.rho_betas[t]) / (
            1 - self.rho_alphas_bar[t]
        )
        c_t = ((self.rho_alphas[t]).sqrt() * (1 - self.rho_alphas_bar[t - 1])) / (
            1 - self.rho_alphas_bar[t]
        )
        mu = c_0 * x_0 + c_t * x_t
        mask = t == 0
        mu[mask] = x_0[mask]

        return mu

    def get_rho_t(
        self, rho: torch.FloatTensor, t: torch.LongTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        x_0 = self.rho_to_vect(rho)

        epsilon = torch.randn_like(x_0)
        epsilon[:, :3] = 0.0
        epsilon[:, -1] *= 0.25

        x_t = (
            x_0 * self.rho_alphas_bar[t].sqrt()[:, None]
            + epsilon * (1 - self.rho_alphas_bar[t]).sqrt()[:, None]
        )

        # mu_t = self.get_mu_rho_t(x_t, x_0, t)

        return self.vect_to_rho(x_t)  # , self.vect_to_rho(mu_t)

    def get_x_t(
        self,
        x: torch.FloatTensor,
        t: torch.LongTensor,
        num_atoms: torch.LongTensor,
    ) -> torch.FloatTensor:
        batch = torch.arange(
            num_atoms.shape[0], dtype=torch.long, device=x.device
        ).repeat_interleave(num_atoms)
        epsilon = torch.randn_like(x)
        t_batch = t.repeat_interleave(num_atoms, dim=0)

        traj = epsilon * (1 - self.x_alphas_bar[t_batch]).sqrt()[:, None]
        traj -= scatter_mean(traj, batch, dim=0, dim_size=num_atoms.shape[0])[batch]

        return (x + traj) % 1.0

    def get_density(
        self,
        rho: torch.FloatTensor,
        z: torch.LongTensor,
        num_atoms: torch.LongTensor,
    ) -> torch.FloatTensor:
        batch = torch.arange(num_atoms.shape[0], device=rho.device).repeat_interleave(
            num_atoms
        )
        masses = scatter_add(
            self.atomic_masses[z], batch, dim=0, dim_size=num_atoms.shape[0]
        )
        return 1.66054 * masses / rho.det()

    def limit_density(
        self,
        rho: torch.FloatTensor,
        z: torch.LongTensor,
        num_atoms: torch.LongTensor,
        rho_backup: torch.FloatTensor,
    ) -> torch.FloatTensor:
        densities = self.get_density(rho, z, num_atoms)

        mask = (densities < self.density_min) | (densities > self.density_max)
        if any(mask):
            rho[mask] = rho_backup[mask]
            warnings.warn("[Langevin dynamics] density constraint reached")

        return rho

    def forward(
        self,
        x: torch.FloatTensor,
        z: torch.LongTensor,
        num_atoms: torch.LongTensor,
    ) -> torch.FloatTensor:
        eye = (
            torch.eye(3, device=num_atoms.device)
            .unsqueeze(0)
            .repeat(num_atoms.shape[0], 1, 1)
        )

        geometry = Geometry(
            eye,
            num_atoms,
            x,
            knn=self.knn,
            triplets=False,
            symetric=True,
            compute_reverse_idx=True,
        )

        _, x_prime, x_traj, rho_prime = self.gnn(z, geometry)

        return x_prime, x_traj, rho_prime

    def get_loss(
        self,
        rho: torch.FloatTensor,
        x: torch.FloatTensor,
        z: torch.LongTensor,
        num_atoms: torch.LongTensor,
        t: torch.LongTensor = None,
        return_data: bool = False,
    ) -> Union[
        Tuple[
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
        ],
        Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor],
    ]:
        if t is None:
            t = torch.randint_like(
                num_atoms, low=0, high=self.x_alphas_bar.shape[0] - 1
            )

        rho_t = self.get_rho_t(rho, t)
        x_t = self.get_x_t(x, t, num_atoms)

        emb = self.t_embedding(t)

        pred_x, pred_traj, pred_rho = self.forward(x_t, z, num_atoms)

        loss_lattice = self.loss_lattice_fn(pred_rho, rho)
        loss_pos = self.loss_pos_fn(rho, x, x_t, pred_traj, num_atoms)
        loss = loss_pos + loss_lattice

        if return_data:
            return (
                loss,
                loss_pos,
                loss_lattice,
                rho_t,
                x_t,
                pred_rho,
                pred_x,
            )

        return loss, loss_pos, loss_lattice

    def sample(
        self,
        x_t: torch.FloatTensor,
        rho_t: torch.FloatTensor,
        rho_0: torch.FloatTensor,
        t: int,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        if t == 0:
            return x_t, rho_0

        x_rand = torch.randn_like(x_t)
        rho_rand = F.pad(torch.randn((rho_0.shape[0], 6), device=rho_0.device), (3, 0))
        rho_rand[:, -1] *= 0.25
        x_prev = (x_t + self.x_sigma[t - 1] * x_rand) % 1.0

        rho_t = self.rho_to_vect(rho_t)
        rho_0 = self.rho_to_vect(rho_0)
        rho_t[:, :3] = 0.0
        rho_0[:, :3] = 0.0
        rho_t1 = self.get_mu_rho_t(rho_t, rho_0, t)

        rho_prev = self.vect_to_rho(rho_t1 + self.rho_sigma[t - 1] * rho_rand)

        return x_prev, rho_prev

    @torch.no_grad()
    def sampling(
        self,
        z: torch.LongTensor,
        num_atoms: torch.LongTensor,
        return_history: bool = False,
        verbose: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        rho = self.random_rho_T(num_atoms.shape[0], device=z.device)
        x = torch.rand((z.shape[0], 3), device=z.device)

        rho_history, x_history = [], []
        if return_history:
            rho_history.append(rho)
            x_history.append(x)

        t_list = list(range(self.diffusion_steps - 1, -1, -1))
        if verbose:
            iterator = tqdm.tqdm(t_list, desc="sampling", leave=False)
        else:
            iterator = t_list

        for t in iterator:
            emb = self.t_embedding(torch.full_like(z, fill_value=t))
            pred_x, _, pred_rho = self.gemsnet(x, z, num_atoms, emb)
            x, rho = self.sample(pred_x, rho, pred_rho, t)

            if return_history:
                rho_history.append(rho)
                x_history.append(x)

        if return_history:
            rho = torch.stack(rho_history, dim=0)
            x = torch.stack(x_history, dim=0)

        return rho, x
