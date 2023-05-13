import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean
from torch_ema import ExponentialMovingAverage
import pandas as pd
import tqdm

import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms
from ase.io import write
from ase.spacegroup import crystal

import os
import json
import math
import random
import datetime

from src.utils.scaler import LatticeScaler
from src.utils.data import MP20, Carbon24, Perov5
from src.utils.hparams import Hparams
from src.utils.metrics import get_metrics
from src.model.gemsnet import GemsNetDiffusion
from src.utils.video import make_video
from src.utils.cif import make_cif
from src.loss import OptimalTrajLoss, LatticeParametersLoss


def get_dataloader(path: str, dataset: str, batch_size: int):
    assert dataset in ["mp-20", "carbon-24", "perov-5"]

    dataset_path = os.path.join(path, dataset)
    if dataset == "mp-20":
        train_set = MP20(dataset_path, "train")
        valid_set = MP20(dataset_path, "val")
        test_set = MP20(dataset_path, "test")
    elif dataset == "carbon-24":
        train_set = Carbon24(dataset_path, "train")
        valid_set = Carbon24(dataset_path, "val")
        test_set = Carbon24(dataset_path, "test")
    elif dataset == "perov-5":
        train_set = Perov5(dataset_path, "train")
        valid_set = Perov5(dataset_path, "val")
        test_set = Perov5(dataset_path, "test")

    loader_train = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4
    )
    loader_valid = DataLoader(
        valid_set, batch_size=batch_size, shuffle=True, num_workers=4
    )
    loader_test = DataLoader(
        test_set, batch_size=batch_size, shuffle=True, num_workers=4
    )

    return loader_train, loader_valid, loader_test


@torch.no_grad()
def compute_metrics(model, dataloader, desc_bar):
    list_losses = []
    list_losses_pos = []
    list_losses_lat = []

    list_t = []
    list_rho = []
    list_rho_pred = []
    list_rho_t = []
    list_x = []
    list_x_pred = []
    list_x_t = []
    list_num_atoms = []

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
            loss,
            loss_pos,
            loss_lattice,
            rho_t,
            x_t,
            pred_rho,
            pred_x,
        ) = model.get_loss(
            batch.cell,
            batch.pos,
            batch.z,
            batch.num_atoms,
            t=t[: batch.num_atoms.shape[0]],
            return_data=True,
        )

        list_t.append(t_idx[: batch.num_atoms.shape[0]])
        list_rho.append(batch.cell)
        list_rho_pred.append(pred_rho)
        list_rho_t.append(rho_t)
        list_x.append(batch.pos)
        list_x_pred.append(pred_x)
        list_x_t.append(x_t)
        list_num_atoms.append(batch.num_atoms)

        list_losses.append(loss.item())
        list_losses_pos.append(loss_pos.item())
        list_losses_lat.append(loss_lattice.item())

    loss = {
        "loss": torch.tensor(list_losses).mean().item(),
        "pos": torch.tensor(list_losses_pos).mean().item(),
        "lattice": torch.tensor(list_losses_lat).mean().item(),
    }

    list_t = torch.cat(list_t, dim=0)
    list_rho = torch.cat(list_rho, dim=0)
    list_rho_pred = torch.cat(list_rho_pred, dim=0)
    list_rho_t = torch.cat(list_rho_t, dim=0)
    list_x = torch.cat(list_x, dim=0)
    list_x_pred = torch.cat(list_x_pred, dim=0)
    list_x_t = torch.cat(list_x_t, dim=0)
    list_num_atoms = torch.cat(list_num_atoms, dim=0)

    metrics = get_metrics(
        list_rho,
        list_rho_pred,
        list_x,
        list_x_pred,
        list_num_atoms,
        by_structure=True,
    )
    metrics_gt = get_metrics(
        list_rho,
        list_rho_t,
        list_x,
        list_x_t,
        list_num_atoms,
        by_structure=True,
    )

    metrics["mae_pos_by_t"] = scatter_mean(
        metrics["mae_pos"], list_t, dim=0, dim_size=size
    )
    metrics["mae_lengths_by_t"] = scatter_mean(
        metrics["mae_lengths"].mean(dim=1), list_t, dim=0, dim_size=size
    )
    metrics["mae_angles_by_t"] = scatter_mean(
        metrics["mae_angles"].mean(dim=1), list_t, dim=0, dim_size=size
    )

    metrics_gt["mae_pos_by_t"] = scatter_mean(
        metrics_gt["mae_pos"], list_t, dim=0, dim_size=size
    )
    metrics_gt["mae_lengths_by_t"] = scatter_mean(
        metrics_gt["mae_lengths"].mean(dim=1), list_t, dim=0, dim_size=size
    )
    metrics_gt["mae_angles_by_t"] = scatter_mean(
        metrics_gt["mae_angles"].mean(dim=1), list_t, dim=0, dim_size=size
    )

    t = torch.arange(0, metrics["mae_pos_by_t"].shape[0]) * step

    metrics["t"] = t
    metrics["mae_pos"] = metrics["mae_pos"]
    metrics["mae_lengths"] = metrics["mae_lengths"]
    metrics["mae_angles"] = metrics["mae_angles"]

    metrics_gt["t"] = t
    metrics_gt["mae_pos"] = metrics_gt["mae_pos"]
    metrics_gt["mae_lengths"] = metrics_gt["mae_lengths"]
    metrics_gt["mae_angles"] = metrics_gt["mae_angles"]

    return loss, metrics, metrics_gt


def add_tensorboard(writer, loss, metrics, metrics_gt, path, batch_idx):
    plt.scatter(metrics["t"], metrics["mae_pos_by_t"], label="gnn")
    plt.scatter(metrics_gt["t"], metrics_gt["mae_pos_by_t"], label="no action")
    plt.legend()
    writer.add_figure(f"{path}/mae_pos", plt.gcf(), batch_idx)
    plt.close()
    plt.scatter(metrics["t"], metrics["mae_lengths_by_t"], label="gnn")
    plt.scatter(metrics_gt["t"], metrics_gt["mae_lengths_by_t"], label="no action")
    plt.legend()
    writer.add_figure(f"{path}/mae_lengths", plt.gcf(), batch_idx)
    plt.close()
    plt.scatter(metrics["t"], metrics["mae_angles_by_t"], label="gnn")
    plt.scatter(metrics_gt["t"], metrics_gt["mae_angles_by_t"], label="no action")
    plt.legend()
    writer.add_figure(f"{path}/mae_angles", plt.gcf(), batch_idx)
    plt.close()

    writer.add_scalar(f"{path}/mae_pos", metrics["mae_pos"].mean(), batch_idx)
    writer.add_scalar(f"{path}/mae_lengths", metrics["mae_lengths"].mean(), batch_idx)
    writer.add_scalar(f"{path}/mae_angles", metrics["mae_angles"].mean(), batch_idx)

    writer.add_scalar(f"{path}/loss", loss["loss"], batch_idx)
    writer.add_scalar(f"{path}/loss_pos", loss["pos"], batch_idx)
    writer.add_scalar(f"{path}/loss_lattice", loss["lattice"], batch_idx)


if __name__ == "__main__":
    import argparse

    from torch.utils.tensorboard import SummaryWriter

    parser = argparse.ArgumentParser(description="train denoising model")
    parser.add_argument("--hparams", "-H", default=None, help="json file")
    parser.add_argument("--logs", "-l", default="./runs/diffusion")
    parser.add_argument("--dataset", "-D", default="mp-20")
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
        features=hparams.features,
        knn=hparams.knn,
        num_blocks=hparams.layers,
        vector_fields=hparams.vector_fields,
        x_betas=hparams.x_betas,
        rho_betas=hparams.rho_betas,
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clipping)
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
            loss, metrics, metrics_gt = compute_metrics(
                model, loader_valid, "validation"
            )
            add_tensorboard(writer, loss, metrics, metrics_gt, "valid", batch_idx)

            if loss["loss"] < best_val:
                torch.save(model.state_dict(), os.path.join(log_dir, "best.pt"))
                loss["loss"] = best_val

    for batch in loader_test:
        batch = batch.to(device)

        limit_batch_size = 16
        batch.num_atoms = batch.num_atoms[:limit_batch_size]
        batch.cell = batch.cell[:limit_batch_size]
        max_atoms = batch.num_atoms.sum()
        batch.pos = batch.pos[:max_atoms]
        batch.z = batch.z[:max_atoms]
        break

    rho, x = model.sampling(batch.z, batch.num_atoms, return_history=True, verbose=True)

    cif = make_cif(rho[-1], x[-1], batch.z, batch.num_atoms)

    with open(os.path.join(log_dir, "sampling.cif"), "w") as fp:
        fp.write(cif)

    video_tensor = make_video(rho, x, batch.z, batch.num_atoms, step=32)

    writer.add_video("sampling", video_tensor)

    loss, metrics, metrics_gt = compute_metrics(model, loader_test, "test")
    add_tensorboard(writer, loss, metrics, metrics_gt, "test", batch_idx)

    metrics = {
        "loss": loss["loss"],
        "loss_pos": loss["pos"],
        "loss_lattice": loss["lattice"],
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
