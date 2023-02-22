import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
import pandas as pd
import tqdm

import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms
from ase.io import write
from ase.spacegroup import crystal

from utils.materials_project import MaterialsProject
from model.cry.attention import Denoiser
from loss.min_distance_loss import MinDistanceLoss
from loss.periodic_relative_loss import PeriodicRelativeLoss
from loss.relative_loss import RelativeLoss

batch_size = 64

dataset = MaterialsProject("./data/mp", pre_filter=lambda x: x.num_atoms <= 32)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print("load", len(dataset), "structures")

model = Denoiser(256, 8, 4, hidden_dim=128).to("cuda")
loss_fn = MinDistanceLoss(center=True).to("cuda")
# loss_relative = PeriodicRelativeLoss().to("cuda")
loss_relative = RelativeLoss().to("cuda")
# loss_fn = nn.BCEWithLogitsLoss()

opt = optim.Adam(model.parameters(), lr=1e-3)

mu, sigma = None, None

logs_path = "./logs_gemnet"
os.makedirs(logs_path, exist_ok=True)


def save_grad(batch, model, filename=None):
    batch = batch.clone().cpu().detach()

    data = {
        "batch": {
            "cell": batch.cell,
            "x": batch.pos,
            "x_thild": batch.x_thild,
            "z": batch.z,
            "num_atoms": batch.num_atoms,
        },
        "model": {
            k: (
                {"weigths": v.data.clone()}
                if v.grad is None
                else {"weigths": v.data.clone(), "grad": v.grad.clone()}
            )
            for k, v in model.named_parameters()
        },
    }

    if filename is not None:
        path, _ = os.path.split(os.path.abspath(filename))
        os.makedirs(path, exist_ok=True)
        torch.save(data, filename)

    return data


def grad_mask(batch, model, idx):
    opt.zero_grad()
    mask = batch.batch == idx

    batch_masked = batch.clone()

    batch_masked.cell = batch_masked.cell[idx : idx + 1]
    batch_masked.material_id = batch_masked.material_id[idx : idx + 1]
    batch_masked.num_atoms = batch_masked.num_atoms[idx : idx + 1]
    batch_masked.y = batch_masked.y[idx : idx + 1]
    batch_masked.pos = batch_masked.pos[mask]
    batch_masked.z = batch_masked.z[mask]
    batch_masked.batch = batch_masked.batch[mask]
    batch_masked.x_thild = batch_masked.x_thild[mask]

    x_prime, _ = model.forward(
        batch_masked.cell,
        batch_masked.pos,
        x_thild,
        batch_masked.z,
        batch_masked.num_atoms,
    )
    loss, detailed = loss_relative(batch.cell, batch.pos, x_prime, batch.num_atoms)
    loss.backward()
    return model.actions[0].pos[2].weight.grad


def save_snapshot(batch, model, filename, n=(6, 2), figsize=(30, 30)):
    _, x_traj = model.forward(
        batch.cell, batch.pos, batch.x_thild, batch.z, batch.num_atoms
    )
    _, ax = plt.subplots(n[0], n[1] * 3, figsize=figsize)

    batch_cpu = batch.clone()
    batch_cpu = batch_cpu.cpu().detach()

    for i in range(n[0]):
        for j in range(n[1]):
            idx = j + i * n[1]
            mask = batch_cpu.batch == idx

            path, _ = os.path.splitext(os.path.abspath(filename))
            path = os.path.join(path, f"{idx}")
            os.makedirs(path, exist_ok=True)

            atoms = crystal(
                batch_cpu.z[mask].numpy(),
                basis=batch_cpu.x_thild[mask].numpy(),
                cell=batch_cpu.cell[idx].numpy(),
            )
            write(os.path.join(path, f"noisy.cif"), atoms)
            plot_atoms(atoms, ax[i][j * 3 + 0], radii=0.3)

            atoms.set_scaled_positions(x_traj[mask].cpu().detach().numpy())
            write(os.path.join(path, f"denoised.cif"), atoms)
            plot_atoms(atoms, ax[i][j * 3 + 1], radii=0.3)

            atoms.set_scaled_positions(batch_cpu.pos[mask].numpy())
            write(os.path.join(path, f"original.cif"), atoms)
            plot_atoms(atoms, ax[i][j * 3 + 2], radii=0.3)
    plt.savefig(filename)
    plt.close()

    for idx in range(n[0] * n[1]):
        ax = plt.figure().add_subplot(projection="3d")

        mask = batch_cpu.batch == idx

        x, y, z = batch_cpu.cell[idx] @ batch_cpu.x_thild[mask].t()
        x_t, y_t, z_t = batch_cpu.cell[idx] @ x_traj[mask].cpu().detach().t()

        ax.scatter(x, y, z, c=batch_cpu.z[mask])
        for a, b, c, d, e, f in zip(x, y, z, x_t, y_t, z_t):
            ax.plot([a, d], [b, e], [c, f], c="black")

        plt.savefig(f"sample_{idx}.png")
        plt.close()


for batch in loader:
    batch_valid = batch.to("cuda")
    x_thild = (batch.pos + 0.1 * torch.randn_like(batch.pos)) % 1.0
    batch_valid.x_thild = x_thild
    break

history = []
logs = {"batch": [], "loss": []}
for name, param in model.named_parameters():
    if param.requires_grad:
        logs[f"{name}.mean"] = []
        logs[f"{name}.std"] = []
        logs[f"{name}.grad.mean"] = []
        logs[f"{name}.grad.std"] = []

batch_idx = 0
for epoch in tqdm.tqdm(range(256), leave=True, position=0):
    losses, losses_rec, losses_rel = [], [], []
    it = tqdm.tqdm(loader, leave=False, position=1)

    for batch in it:
        batch = batch.to("cuda")

        opt.zero_grad()

        x_thild = (batch.pos + 0.1 * torch.randn_like(batch.pos)) % 1.0
        batch.x_thild = x_thild

        x_prime = model.forward(
            batch.cell, batch.pos, x_thild, batch.z, batch.num_atoms
        )

        # if mu is None:
        #    mu, sigma = mask.float().mean(), mask.float().std()

        # loss = F.mse_loss(pred_edges, (mask.float() - mu) / sigma)
        loss_rec = loss_fn(batch.cell, batch.pos, x_prime, batch.num_atoms)
        loss, detailed = loss_relative(batch.cell, batch.pos, x_prime, batch.num_atoms)
        loss = loss_rec
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # save_grad(batch, model, os.path.join(logs_path, f"{batch_idx}.pt"))

        """
        history.append(save_grad(batch, model))
        if len(history) > 32:
            history = history[-32:]

        # actions.0.pos.2.weight
        for action in model.actions:
            if (action.pos[2].weight.grad is not None) and (
                action.pos[2].weight.grad.mean() > 1.0
            ):
                import matplotlib.pyplot as plt

                plt.hist(detailed.clone().detach().cpu().numpy(), bins=32)
                plt.savefig(os.path.join(logs_path, f"hist_loss.png"))
                save_snapshot(batch, model, "test.png")

                torch.save(history, "history.pt")

                breakpoint()
                break
        """

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # loss_zero = loss_fn(batch.cell, batch.pos, x_thild, batch.num_atoms).item()
        losses.append(loss.item())

        # it.set_description(f"loss: {loss.item():.3f}/{loss_zero:.3f}")
        it.set_description(f"loss: {loss.item():.3f} rec={loss_rec.item():.3f}")

        if (batch_idx % 16) == 0:
            logs["batch"].append(batch_idx)
            logs["loss"].append(torch.tensor(losses).mean().item())

            for name, param in model.named_parameters():
                if param.requires_grad:
                    logs[f"{name}.mean"].append(param.data.mean().item())
                    logs[f"{name}.std"].append(param.data.std().item())
                    if param.grad is not None:
                        logs[f"{name}.grad.mean"].append(param.grad.mean().item())
                        logs[f"{name}.grad.std"].append(param.grad.std().item())
                    else:
                        logs[f"{name}.grad.mean"].append(0.0)
                        logs[f"{name}.grad.std"].append(0.0)

            pd.DataFrame(logs).set_index("batch").to_csv(
                os.path.join(logs_path, "loss.csv")
            )

            plt.close()
            plt.plot(logs["batch"], logs["loss"])
            plt.savefig(os.path.join(logs_path, "loss.png"))

            # save_snapshot(batch_valid, model, os.path.join(logs_path, f"{batch_idx}.png"))

        batch_idx += 1
