import torch
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms
from ase.io import write
from ase.spacegroup import crystal

import os

def save_snapshot(batch, model, filename, n=(6, 2), figsize=(30, 30),noise_pos=0.05):
    x_thild = (batch.pos + noise_pos * torch.randn_like(batch.pos)) % 1.0
    batch.x_thild = x_thild
    eye = torch.eye(3, device=batch.pos.device).unsqueeze(0).repeat(batch.cell.shape[0], 1, 1)

    x_traj,rho_prime = model.forward(eye, x_thild, batch.z, batch.num_atoms)
    
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
