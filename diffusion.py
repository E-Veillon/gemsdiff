import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from torch_ema import ExponentialMovingAverage
import pandas as pd
import tqdm

import matplotlib.pyplot as plt

import os
import json
import random
import datetime

from src.utils.scaler import LatticeScaler
from src.utils.data import MP, OQMD, StructuresSampler
from src.utils.hparams import Hparams
from src.utils.metrics import compute_metrics
from src.model.gemsnet import GemsNetDiffusion
from src.utils.video import make_video
from src.utils.cif import make_cif


def get_dataloader(path: str, dataset: str, batch_size: int, _old: bool = True):
    """
    Load and split dataset into training, validation and test DataLoaders.

    Parameters:
        path (str):         Path to directory containing the dataset file.

        dataset (str):      Name of the dataset to use. Supports "mp" and "oqmd".

        batch_size (int):   Number of data to put in each DataLoader batch.

        _old (bool):        Dev flag to switch easily between two different
                            implementations of this function.
    
    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: Train, Validation and Test DataLoaders.
    """
    assert dataset in ["mp", "oqmd"]

    dataset_path = os.path.join(path, dataset)
    if _old:
        if dataset == "mp":
            data = MP(dataset_path)
            gen = torch.Generator().manual_seed(42)
            train_set, valid_set, test_set = random_split(
                data, [78600, 4367, 4367], generator=gen
            )
        elif dataset == "oqmd":
            data = OQMD(dataset_path)
            gen = torch.Generator().manual_seed(42)
            train_set, valid_set, test_set = random_split(
                data, [199686, 11094, 11094], generator=gen
            )
    else:
        data = MP(dataset_path) if dataset == "mp" else OQMD(dataset_path)
        n_data = len(data)
        n_train = round(n_data * 0.9)
        n_valid = n_test = round(n_data * 0.05)

        # Handle rounding error
        remainder = n_data - (n_train + n_valid + n_test)

        if remainder == 1: # Too much floor rounding
            n_train += 1

        elif remainder == -1: # Too much ceil rounding
            n_train -= 1

        assert n_data == n_train + n_valid + n_test

        # Split dataset
        gen = torch.Generator().manual_seed(42)
        train_set, valid_set, test_set = random_split(
                data, [n_train, n_valid, n_test], generator=gen
        )

    loader_train = DataLoader(
        train_set,
        num_workers=4,
        batch_sampler=StructuresSampler(train_set, max_atoms=batch_size, shuffle=True),
    )
    loader_valid = DataLoader(valid_set, batch_size=64, num_workers=4)
    loader_test = DataLoader(test_set, batch_size=64, num_workers=4)

    return loader_train, loader_valid, loader_test


def add_tensorboard(writer, metrics, path, batch_idx):
    """Add models metrics for tensorboard plotting."""
    plt.scatter(metrics["t"], metrics["mae_pos_by_t"], label="gnn")
    plt.scatter(metrics["t"], metrics["mae_pos_diff_by_t"], label="no action")
    plt.legend()
    writer.add_figure(f"{path}/mae_pos", plt.gcf(), batch_idx)
    plt.close()

    writer.add_scalar(f"{path}/mae_pos", metrics["mae_pos"].mean(), batch_idx)
    writer.add_scalar(f"{path}/mae_lengths", metrics["mae_lengths"].mean(), batch_idx)
    writer.add_scalar(f"{path}/mae_angles", metrics["mae_angles"].mean(), batch_idx)

    writer.add_scalar(
        f"{path}/loss", metrics["loss_pos"] + metrics["loss_lattice"], batch_idx
    )
    writer.add_scalar(f"{path}/loss_pos", metrics["loss_pos"], batch_idx)
    writer.add_scalar(f"{path}/loss_lattice", metrics["loss_lattice"], batch_idx)


if __name__ == "__main__":
    import argparse

    from torch.utils.tensorboard import SummaryWriter

    parser = argparse.ArgumentParser(description="train denoising model")
    parser.add_argument("--hparams", "-H", default=None, help="json file")
    parser.add_argument("--logs", "-l", default="./runs/diffusion")
    parser.add_argument("--dataset", "-D", default="oqmd")
    parser.add_argument("--dataset-path", "-dp", default="./data")
    parser.add_argument("--device", "-d", default="cuda")
    parser.add_argument("--threads", "-t", type=int, default=8)

    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)

    # run name
    tday = datetime.datetime.now()
    run_name = tday.strftime(
        f"training_%Y_%m_%d_%H_%M_%S_{args.dataset}_{random.randint(0,1000):<03d}"
    )
    print("run name:", run_name)

    log_dir = os.path.join(args.logs, run_name)

    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir, flush_secs=3)

    # basic setup
    device = args.device

    hparams = Hparams()
    if args.hparams is not None:
        hparams.from_json(args.hparams)

    with open(os.path.join(log_dir, "hparams.json"), "w") as fp:
        json.dump(hparams.dict(), fp, indent=4)

    print("hparams:")
    print(json.dumps(hparams.dict(), indent=4))

    loader_train, loader_valid, loader_test = get_dataloader(
        args.dataset_path, args.dataset, hparams.batch_size
    )

    scaler = LatticeScaler().to(device)
    scaler.fit(loader_train)

    model = GemsNetDiffusion(
        lattice_scaler=scaler,
        knn=hparams.knn,
        num_blocks=hparams.layers,
        x_betas=hparams.x_betas,
        diffusion_steps=hparams.diffusion_steps,
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=hparams.lr, betas=(hparams.beta1, 0.999))
    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)

    logs = {"batch": [], "loss": [], "loss_pos": [], "loss_lat": []}

    best_val = float("inf")

    batch_idx = 0
    snapshot_idx = 0
    for epoch in tqdm.tqdm(range(hparams.epochs), leave=True, position=0):
        losses, losses_pos, losses_lat = [], [], []

        it = tqdm.tqdm(loader_train, leave=False, position=1)

        for batch in it:
            batch = batch.to(device)

            opt.zero_grad()

            loss, loss_pos, loss_lat = model.get_loss(
                batch.cell, batch.pos, batch.z, batch.num_atoms
            )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clipping)
            opt.step()
            ema.update()

            losses.append(loss.item())
            losses_pos.append(loss_pos.item())
            losses_lat.append(loss_lat.item())

            it.set_description(
                f"loss: {loss.item():.3f} atomic pos={loss_pos.item():.3f} lattice={loss_lat.item():.3f}"
            )

            batch_idx += 1

        losses = torch.tensor(losses).mean().item()
        losses_pos = torch.tensor(losses_pos).mean().item()
        losses_lat = torch.tensor(losses_lat).mean().item()

        writer.add_scalar("train/loss", losses, batch_idx)
        writer.add_scalar("train/loss_pos", losses_pos, batch_idx)
        writer.add_scalar("train/loss_lattice", losses_lat, batch_idx)

        logs["batch"].append(batch_idx)
        logs["loss"].append(losses)
        logs["loss_pos"].append(losses_pos)
        logs["loss_lat"].append(losses_lat)

        pd.DataFrame(logs).set_index("batch").to_csv(os.path.join(log_dir, "loss.csv"))

        with ema.average_parameters():
            metrics = compute_metrics(model, loader_valid, "validation", device)
            add_tensorboard(writer, metrics, "valid", batch_idx)

            total_loss = metrics["loss_pos"] + metrics["loss_lattice"]
            if total_loss < best_val:
                torch.save(model.state_dict(), os.path.join(log_dir, "best.pt"))
                best_val = total_loss

    for batch in loader_test:
        batch = batch.to(device)
        break

    limit_batch_size = 16
    batch.num_atoms = batch.num_atoms[:limit_batch_size]
    batch.cell = batch.cell[:limit_batch_size]
    max_atoms = batch.num_atoms.sum()
    batch.pos = batch.pos[:max_atoms]
    batch.z = batch.z[:max_atoms]

    rho, x = model.sampling(batch.z, batch.num_atoms, return_history=True, verbose=True)

    cif = make_cif((rho[0][-1], rho[1][-1]), x[-1], batch.z, batch.num_atoms)

    with open(os.path.join(log_dir, "sampling.cif"), "w") as fp:
        fp.write(cif)

    # video_tensor = make_video(rho, x, batch.z, batch.num_atoms, step=32)

    # writer.add_video("sampling", video_tensor)

    metrics = compute_metrics(model, loader_test, "test", device)
    add_tensorboard(writer, metrics, "test", batch_idx)

    metrics = {
        "loss": metrics["loss_pos"] + metrics["loss_lattice"],
        "loss_pos": metrics["loss_pos"],
        "loss_lattice": metrics["loss_lattice"],
        "mae_pos": metrics["mae_pos"].mean().item(),
        "mae_lengths": metrics["mae_lengths"].mean().item(),
        "mae_angles": metrics["mae_angles"].mean().item(),
    }

    with open(os.path.join(log_dir, "metrics.json"), "w") as fp:
        json.dump(metrics, fp, indent=4)

    print("\nmetrics:")
    print(json.dumps(metrics, indent=4))

    writer.add_hparams(hparams.dict(), metrics)

    writer.close()
