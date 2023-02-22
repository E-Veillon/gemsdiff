import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import QM9

import tqdm

from model.mol.valance import Denoise

dataset = QM9("./data/qm9")
loader = DataLoader(dataset, batch_size=256, shuffle=True)


def test(loader, model):
    model = model.to("cuda")
    loss_fn = nn.MSELoss()

    opt = optim.Adam(model.parameters(), lr=1e-3)

    first = None

    for _ in range(2):
        it = tqdm.tqdm(loader, leave=False)
        for batch in it:
            batch = batch.to("cuda")
            num_atoms = batch.ptr[1:] - batch.ptr[:-1]

            noisy_x = batch.pos + 0.1 * torch.randn_like(batch.pos)

            opt.zero_grad()

            x_prime = model(batch.pos, noisy_x, batch.z, num_atoms, batch.edge_index)

            d_prime_ij = (
                x_prime[batch.edge_index[0]] - x_prime[batch.edge_index[1]]
            ).norm(dim=1)
            d_ij = (
                batch.pos[batch.edge_index[0]] - batch.pos[batch.edge_index[1]]
            ).norm(dim=1)
            loss = loss_fn(d_prime_ij, d_ij)
            loss.backward()

            loss_zero = loss_fn(noisy_x, batch.pos).item()

            opt.step()

            if first is None:
                first = loss.item()

            it.set_description(f"loss: {loss.item():.3f}/{loss_zero:.3f} ({first:.3f})")

    with torch.no_grad():
        losses = []
        for batch in tqdm.tqdm(loader, desc="avg", leave=False):
            batch = batch.to("cuda")
            num_atoms = batch.ptr[1:] - batch.ptr[:-1]

            noisy_x = batch.pos + 0.1 * torch.randn_like(batch.pos)

            x_prime = model(batch.pos, noisy_x, batch.z, num_atoms, batch.edge_index)

            d_prime_ij = (
                x_prime[batch.edge_index[0]] - x_prime[batch.edge_index[1]]
            ).norm(dim=1)
            d_ij = (
                batch.pos[batch.edge_index[0]] - batch.pos[batch.edge_index[1]]
            ).norm(dim=1)
            loss = loss_fn(d_prime_ij, d_ij)
            losses.append(loss.item())
    return torch.tensor(losses).mean()


pm = "\u00b1"
n = 8

metric = []
for _ in range(n):
    metric.append(test(loader, Denoise(128, 4, 64, gen_graph=False)))
metric = torch.tensor(metric)
print(
    f"valance (graph gen off): {metric.mean().item():.5f}{pm}{metric.std().item():.5f}"
)

metric = []
for _ in range(n):
    metric.append(
        test(
            loader,
            Denoise(
                128, 4, 64, gen_graph=True, covalent=True, from_original_geometry=True
            ),
        )
    )
metric = torch.tensor(metric)
print(
    f"valance (graph gen on, covalent): {metric.mean().item():.5f}{pm}{metric.std().item():.5f}"
)

metric = []
for _ in range(n):
    metric.append(
        test(
            loader,
            Denoise(
                128,
                4,
                64,
                gen_graph=True,
                covalent=True,
                from_original_geometry=True,
                attention=True,
            ),
        )
    )
metric = torch.tensor(metric)
print(
    f"valance (graph gen on, covalent, attention): {metric.mean().item():.5f}{pm}{metric.std().item():.5f}"
)

metric = []
for _ in range(n):
    metric.append(
        test(
            loader,
            Denoise(
                128,
                4,
                64,
                gen_graph=True,
                covalent=True,
                from_original_geometry=True,
                attention=False,
                normalize=True,
            ),
        )
    )
metric = torch.tensor(metric)
print(
    f"valance (graph gen on, covalent, normalized): {metric.mean().item():.5f}{pm}{metric.std().item():.5f}"
)

metric = []
for _ in range(n):
    metric.append(
        test(
            loader,
            Denoise(
                128,
                4,
                64,
                gen_graph=True,
                covalent=True,
                from_original_geometry=True,
                attention=True,
                normalize=True,
            ),
        )
    )
metric = torch.tensor(metric)
print(
    f"valance (graph gen on, covalent, attention, normalized): {metric.mean().item():.5f}{pm}{metric.std().item():.5f}"
)
