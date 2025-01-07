import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
import tqdm

from typing import Tuple, Union

from src.utils.scaler import LatticeScaler
from src.utils.geometry import Geometry
from src.loss import OptimalTrajLoss, LatticeParametersLoss
from .gemsnet import GemsNetT


class GemsNetDiffusion(nn.Module):
    def __init__(
        self,
        lattice_scaler: LatticeScaler,
        features: int = 256,
        knn: int = 32,
        num_blocks: int = 3,
        x_betas: Tuple[float, float] = (1e-6, 2e-4),
        diffusion_steps: int = 100,
    ):
        super().__init__()

        # energy_targets
        self.knn = knn
        self.gnn = GemsNetT(
            features,
            num_blocks=num_blocks,
            energy_targets=1,
            compute_energy=False,
            compute_forces=True,
            compute_stress=True,
        )

        self.loss_lattice_fn = LatticeParametersLoss(lattice_scaler=lattice_scaler)
        self.loss_pos_fn = OptimalTrajLoss(center=True, euclidian=True, distance="l1")

        self.x_betas = nn.Parameter(
            torch.linspace(x_betas[0], x_betas[1], diffusion_steps),
            requires_grad=False,
        )
        self.x_sigma = nn.Parameter(self.x_betas.sqrt(), requires_grad=False)
        self.x_alphas = nn.Parameter(1 - self.x_betas, requires_grad=False)
        self.x_alphas_bar = nn.Parameter(
            torch.cumprod(self.x_alphas, dim=0), requires_grad=False
        )

    @property
    def diffusion_steps(self) -> int:
        return self.x_betas.shape[0]

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
        ],
        Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor],
    ]:
        if t is None:
            t = torch.randint_like(
                num_atoms, low=0, high=self.x_alphas_bar.shape[0] - 1
            )

        x_t = self.get_x_t(x, t, num_atoms)

        pred_x, pred_traj, pred_rho = self.forward(x_t, z, num_atoms)

        loss_lattice = self.loss_lattice_fn(pred_rho, rho)
        loss_pos = self.loss_pos_fn(rho, x, x_t, pred_traj, num_atoms)
        loss = loss_pos + loss_lattice

        if return_data:
            return (
                loss,
                loss_pos,
                loss_lattice,
                x_t,
                self.lattice_scaler.denormalise(*pred_rho),
                pred_x,
            )

        return loss, loss_pos, loss_lattice

    def sample(
        self,
        x_t: torch.FloatTensor,
        t: int,
    ) -> torch.FloatTensor:
        if t == 0:
            return x_t

        x_rand = torch.randn_like(x_t)
        x_prev = (x_t + self.x_sigma[t - 1] * x_rand) % 1.0

        return x_prev

    @torch.no_grad()
    def sampling(
        self,
        z: torch.LongTensor,
        num_atoms: torch.LongTensor,
        return_history: bool = False,
        verbose: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        x = torch.rand((z.shape[0], 3), device=z.device)

        rho_history, x_history = [], []

        t_list = list(range(self.diffusion_steps - 1, -1, -1))
        if verbose:
            iterator = tqdm.tqdm(t_list, desc="sampling", leave=False)
        else:
            iterator = t_list

        for t in iterator:
            pred_x, _, rho = self.forward(x, z, num_atoms)
            x = self.sample(pred_x, t)

            if return_history:
                lengths, angles = self.lattice_scaler.denormalise(*pred_rho)
                rho_history_lengths.append(lengths)
                rho_history_angles.append(angles)
                x_history.append(x)

        if return_history:
            rho_lengths = torch.stack(rho_history_lengths, dim=0)
            rho_angles = torch.stack(rho_history_angles, dim=0)
            x = torch.stack(x_history, dim=0)

        return (rho_lengths, rho_angles), x
