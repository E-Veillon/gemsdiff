import torch
from torch_scatter import scatter_mean
import tqdm

from typing import Dict

from src.utils.scaler import LatticeScaler


def push(history: Dict[str, torch.Tensor], **kwargs):
    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                value = value.unsqueeze(0)

            if key in history:
                history[key] = torch.cat((history[key], value), dim=0)
            else:
                history[key] = value
        elif isinstance(value, tuple) and all(
            [isinstance(v, torch.Tensor) for v in value]
        ):
            h_values = history.get(key, tuple([None] * len(value)))
            for idx, (h, v) in enumerate(zip(h_values, value)):
                if h is None:
                    h_values[idx] = v
                else:
                    h_values[idx] = torch.cat((h, v), dim=0)
            history[key] = h_values
        else:
            raise Exception("error")


@torch.no_grad()
def get_metric_pos(
    rho: torch.FloatTensor,
    x: torch.FloatTensor,
    x_prime: torch.FloatTensor,
    num_atoms: torch.LongTensor,
    by_structure: bool = False,
) -> torch.FloatTensor:
    offset = torch.tensor(
        [
            [-1, -1, -1],
            [-1, -1, 0],
            [-1, -1, 1],
            [-1, 0, -1],
            [-1, 0, 0],
            [-1, 0, 1],
            [-1, 1, -1],
            [-1, 1, 0],
            [-1, 1, 1],
            [0, -1, -1],
            [0, -1, 0],
            [0, -1, 1],
            [0, 0, -1],
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, -1],
            [0, 1, 0],
            [0, 1, 1],
            [1, -1, -1],
            [1, -1, 0],
            [1, -1, 1],
            [1, 0, -1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, -1],
            [1, 1, 0],
            [1, 1, 1],
        ]
    )

    struct_idx = torch.arange(rho.shape[0], device=rho.device)
    batch = struct_idx.repeat_interleave(num_atoms)

    euc_x_prime = torch.einsum(rho[batch], [0, 1, 2], x_prime, [0, 2], [0, 1])

    euc_x = torch.einsum(
        rho[batch],
        [0, 2, 3],
        x[:, None, :] + offset[None, :, :],
        [0, 1, 3],
        [0, 1, 2],
    )

    min_idx = (euc_x_prime[:, None] - euc_x).norm(dim=2).argmin(dim=1)

    error_pos = (x_prime - (x + offset[min_idx])).norm(dim=1)
    if by_structure:
        mae_pos = scatter_mean(
            error_pos, batch, dim=0, dim_size=num_atoms.shape[0]
        ).detach()
    else:
        mae_pos = error_pos.mean().detach()

    return mae_pos


@torch.no_grad()
def get_metrics(
    history: Dict[str, torch.Tensor],
    by_structure: bool = False,
) -> Dict[str, torch.FloatTensor]:
    rho = history["rho"].cpu()
    rho_prime = history["rho_pred"].cpu()
    x = history["x"].cpu() % 1.0
    x_prime = history["x_pred"].cpu() % 1.0
    x_t = history["x_t"].cpu() % 1.0
    num_atoms = history["num_atoms"].cpu()

    pos_mae_gnn = get_metric_pos(rho, x, x_prime, num_atoms, by_structure=by_structure)
    pos_mae_diff = get_metric_pos(rho, x, x_t, num_atoms, by_structure=by_structure)

    rho_lengths, rho_angles = LatticeScaler.get_lattices_parameters(rho)
    rho_prime_lengths, rho_prime_angles = LatticeScaler.get_lattices_parameters(
        rho_prime
    )

    error_lengths = (rho_lengths - rho_prime_lengths).abs()
    error_angles = (rho_angles - rho_prime_angles).abs()
    if by_structure:
        mae_lengths = error_lengths.detach()
        mae_angles = error_angles.detach()
    else:
        mae_lengths = error_lengths.mean().detach()
        mae_angles = error_angles.mean().detach()

    return {
        "loss_pos": history["loss_pos"].mean().item(),
        "loss_lattice": history["loss_lattice"].mean().item(),
        "mae_pos": pos_mae_gnn,
        "mae_pos_diff": pos_mae_diff,
        "mae_lengths": mae_lengths,
        "mae_angles": mae_angles,
    }


@torch.no_grad()
def compute_metrics(model, dataloader, desc_bar, device):
    history = {}

    step = model.diffusion_steps // 32
    N = dataloader.batch_size * step // model.diffusion_steps
    t = torch.arange(0, model.diffusion_steps, step)
    size = t.shape[0]
    t_idx = torch.arange(0, size)
    t = t.repeat(N + 1).to(device)
    t_idx = t_idx.repeat(N + 1)

    for batch in tqdm.tqdm(dataloader, leave=False, position=1, desc=desc_bar):
        batch = batch.to(device)

        (
            _,
            loss_pos,
            _,
            x_t,
            _,
            x_pred,
        ) = model.get_loss(
            batch.cell,
            batch.pos,
            batch.z,
            batch.num_atoms,
            t=t[: batch.num_atoms.shape[0]],
            return_data=True,
        )

        (
            _,
            _,
            loss_lattice,
            _,
            rho_pred,
            _,
        ) = model.get_loss(
            batch.cell,
            batch.pos,
            batch.z,
            batch.num_atoms,
            t=torch.full_like(t[: batch.num_atoms.shape[0]], fill_value=0),
            return_data=True,
        )

        push(
            history,
            t=t_idx[: batch.num_atoms.shape[0]],
            rho=batch.cell,
            rho_pred=rho_pred,
            x=batch.pos,
            x_pred=x_pred,
            x_t=x_t,
            num_atoms=batch.num_atoms,
            loss_pos=loss_pos,
            loss_lattice=loss_lattice,
        )

    metrics = get_metrics(history, by_structure=True)

    metrics["mae_pos_by_t"] = scatter_mean(
        metrics["mae_pos"], history["t"], dim=0, dim_size=size
    )
    metrics["mae_lengths_by_t"] = scatter_mean(
        metrics["mae_lengths"].mean(dim=1), history["t"], dim=0, dim_size=size
    )
    metrics["mae_angles_by_t"] = scatter_mean(
        metrics["mae_angles"].mean(dim=1), history["t"], dim=0, dim_size=size
    )

    metrics["mae_pos_diff_by_t"] = scatter_mean(
        metrics["mae_pos_diff"], history["t"], dim=0, dim_size=size
    )

    metrics["t"] = torch.arange(0, metrics["mae_pos_by_t"].shape[0]) * step

    return metrics
