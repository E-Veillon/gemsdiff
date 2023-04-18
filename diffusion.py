import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean
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
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=hparams.lr, betas=(hparams.beta1, 0.999))

    logs = {"batch": [], "loss": [], "loss_pos": [], "loss_lat": []}

    best_val = float("inf")

    for batch in loader_train:
        batch = batch.to(device)

        limit_batch_size = 8
        batch.num_atoms = batch.num_atoms[:limit_batch_size]
        batch.cell = batch.cell[:limit_batch_size]
        max_atoms = batch.num_atoms.sum()
        batch.pos = batch.pos[:max_atoms]
        batch.z = batch.z[:max_atoms]
        break

    rho, x = model.sampling(batch.z, batch.num_atoms, return_history=True, verbose=True)
    print(rho.shape, x.shape)
    exit(0)

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

        with torch.no_grad():
            valid_losses = []
            valid_losses_pos = []
            valid_losses_lat = []

            valid_t = []
            valid_rho = []
            valid_rho_pred = []
            valid_rho_t = []
            valid_x = []
            valid_x_pred = []
            valid_x_t = []
            valid_num_atoms = []

            step = 32
            N = hparams.diffusion_steps // step
            t = torch.arange(0, hparams.diffusion_steps, step)
            size = t.shape[0]
            t_idx = torch.arange(0, size)
            t = t.repeat(N + 1).to(device)
            t_idx = t_idx.repeat(N + 1)

            for batch in tqdm.tqdm(
                loader_valid, leave=False, position=1, desc="validation"
            ):
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

                valid_t.append(t_idx[: batch.num_atoms.shape[0]])
                valid_rho.append(batch.cell)
                valid_rho_pred.append(pred_rho)
                valid_rho_t.append(rho_t)
                valid_x.append(batch.pos)
                valid_x_pred.append(pred_x)
                valid_x_t.append(x_t)
                valid_num_atoms.append(batch.num_atoms)

                valid_losses.append(loss.item())
                valid_losses_pos.append(loss_pos.item())
                valid_losses_lat.append(loss_lattice.item())

            losses = torch.tensor(losses).mean().item()
            losses_pos = torch.tensor(losses_pos).mean().item()
            losses_lat = torch.tensor(losses_lat).mean().item()

            if losses < best_val:
                best_val = losses
                torch.save(model.state_dict(), os.path.join(log_dir, "best.pt"))

            writer.add_scalar("valid/loss", losses, batch_idx)
            writer.add_scalar("valid/loss_pos", losses_pos, batch_idx)
            writer.add_scalar("valid/loss_lattice", losses_lat, batch_idx)

            valid_t = torch.cat(valid_t, dim=0)
            valid_rho = torch.cat(valid_rho, dim=0)
            valid_rho_pred = torch.cat(valid_rho_pred, dim=0)
            valid_rho_t = torch.cat(valid_rho_t, dim=0)
            valid_x = torch.cat(valid_x, dim=0)
            valid_x_pred = torch.cat(valid_x_pred, dim=0)
            valid_x_t = torch.cat(valid_x_t, dim=0)
            valid_num_atoms = torch.cat(valid_num_atoms, dim=0)

            metrics = get_metrics(
                valid_rho,
                valid_rho_pred,
                valid_x,
                valid_x_pred,
                valid_num_atoms,
                by_structure=True,
            )
            metrics_gt = get_metrics(
                valid_rho,
                valid_rho_t,
                valid_x,
                valid_x_t,
                valid_num_atoms,
                by_structure=True,
            )

            mae_pos_by_t = scatter_mean(
                metrics["mae_pos"], valid_t, dim=0, dim_size=size
            )
            mae_pos_by_t_gt = scatter_mean(
                metrics_gt["mae_pos"], valid_t, dim=0, dim_size=size
            )
            mae_lengths_by_t = scatter_mean(
                metrics["mae_lengths"].mean(dim=1), valid_t, dim=0, dim_size=size
            )
            mae_lengths_by_t_gt = scatter_mean(
                metrics_gt["mae_lengths"].mean(dim=1), valid_t, dim=0, dim_size=size
            )
            mae_angles_by_t = scatter_mean(
                metrics["mae_angles"].mean(dim=1), valid_t, dim=0, dim_size=size
            )
            mae_angles_by_t_gt = scatter_mean(
                metrics_gt["mae_angles"].mean(dim=1), valid_t, dim=0, dim_size=size
            )
            t_range = torch.arange(0, mae_pos_by_t.shape[0]) * step

            plt.scatter(t_range, mae_pos_by_t, label="gnn")
            plt.scatter(t_range, mae_pos_by_t_gt, label="no action")
            plt.legend()
            writer.add_figure("valid/mae_pos", plt.gcf(), batch_idx)
            plt.close()
            plt.scatter(t_range, mae_lengths_by_t, label="gnn")
            plt.scatter(t_range, mae_lengths_by_t_gt, label="no action")
            plt.legend()
            writer.add_figure("valid/mae_lengths", plt.gcf(), batch_idx)
            plt.close()
            plt.scatter(t_range, mae_angles_by_t, label="gnn")
            plt.scatter(t_range, mae_angles_by_t_gt, label="no action")
            plt.legend()
            writer.add_figure("valid/mae_angles", plt.gcf(), batch_idx)
            plt.close()

            writer.add_scalar("valid/mae_pos", metrics["mae_pos"].mean(), batch_idx)
            writer.add_scalar(
                "valid/mae_lengths", metrics["mae_lengths"].mean(), batch_idx
            )
            writer.add_scalar(
                "valid/mae_angles", metrics["mae_angles"].mean(), batch_idx
            )

    with torch.no_grad():
        test_losses = []
        test_losses_pos = []
        test_losses_lat = []

        test_t = []
        test_rho = []
        test_rho_pred = []
        test_x = []
        test_x_pred = []
        test_num_atoms = []

        step = 32
        N = hparams.diffusion_steps // step
        t = torch.arange(0, hparams.diffusion_steps, step).repeat(N + 1).to(device)
        t_idx = torch.arange(0, N).repeat(N + 1)

        for batch in tqdm.tqdm(loader_test, leave=False, position=1, desc="test"):
            batch = batch.to(device)

            loss, loss_pos, loss_lattice, pred_rho, pred_x = model.get_loss(
                batch.cell,
                batch.pos,
                batch.z,
                batch.num_atoms,
                t=t[: batch.num_atoms.shape[0]],
                return_output=True,
            )

            test_t.append(t_idx[: batch.num_atoms.shape[0]])
            test_rho.append(batch.cell)
            test_rho_pred.append(pred_rho)
            test_x.append(batch.pos)
            test_x_pred.append(pred_x)
            test_num_atoms.append(batch.num_atoms)

            test_losses.append(loss.item())
            test_losses_pos.append(loss_pos.item())
            test_losses_lat.append(loss_lattice.item())

        losses = torch.tensor(losses).mean().item()
        losses_pos = torch.tensor(losses_pos).mean().item()
        losses_lat = torch.tensor(losses_lat).mean().item()

        if losses < best_val:
            best_val = losses
            torch.save(model.state_dict(), os.path.join(log_dir, "best.pt"))

        writer.add_scalar("test/loss", losses, batch_idx)
        writer.add_scalar("test/loss_pos", losses_pos, batch_idx)
        writer.add_scalar("test/loss_lattice", losses_lat, batch_idx)

        test_t = torch.cat(test_t, dim=0)
        test_rho = torch.cat(test_rho, dim=0)
        test_rho_pred = torch.cat(test_rho_pred, dim=0)
        test_x = torch.cat(test_x, dim=0)
        test_x_pred = torch.cat(test_x_pred, dim=0)
        test_num_atoms = torch.cat(test_num_atoms, dim=0)

        metrics = get_metrics(
            test_rho,
            test_rho_pred,
            test_x,
            test_x_pred,
            test_num_atoms,
            by_structure=True,
        )

        mae_pos_by_t = scatter_mean(metrics["mae_pos"], test_t, dim=0, dim_size=N)
        mae_lengths_by_t = scatter_mean(
            metrics["mae_lengths"], test_t, dim=0, dim_size=N
        ).mean(dim=1)
        mae_angles_by_t = scatter_mean(
            metrics["mae_angles"], test_t, dim=0, dim_size=N
        ).mean(dim=1)
        t_range = torch.arange(0, mae_pos_by_t.shape[0]) * step

        plt.scatter(t_range, mae_pos_by_t)
        writer.add_figure("test/mae_pos", plt.gcf(), batch_idx)
        plt.close()
        plt.scatter(t_range, mae_lengths_by_t)
        writer.add_figure("test/mae_lengths", plt.gcf(), batch_idx)
        plt.close()
        plt.scatter(t_range, mae_angles_by_t)
        writer.add_figure("test/mae_angles", plt.gcf(), batch_idx)
        plt.close()

        writer.add_scalar("test/mae_pos", metrics["mae_pos"].mean(), batch_idx)
        writer.add_scalar("test/mae_lengths", metrics["mae_lengths"].mean(), batch_idx)
        writer.add_scalar("test/mae_angles", metrics["mae_angles"].mean(), batch_idx)

        metrics = {
            "loss": losses,
            "loss_pos": losses_pos,
            "loss_lattice": losses_lat,
            **{
                k: v.item()
                for k, v in get_metrics(
                    test_rho, test_rho_pred, test_x, test_x_pred, test_num_atoms
                ).items()
            },
        }

        with open(os.path.join(log_dir, "metrics.json"), "w") as fp:
            json.dump(metrics, fp, indent=4)

        print("\nmetrics:")
        print(json.dumps(metrics, indent=4))
        writer.add_scalar("test/loss", losses, batch_idx)
        writer.add_scalar("test/loss_pos", losses_pos, batch_idx)
        writer.add_scalar("test/loss_lattice", losses_lat, batch_idx)

        writer.add_scalar("test/mae_pos", metrics["mae_pos"], batch_idx)
        writer.add_scalar("test/mae_lengths", metrics["mae_lengths"], batch_idx)
        writer.add_scalar("test/mae_angles", metrics["mae_angles"], batch_idx)

        writer.add_hparams(hparams.dict(), metrics)

        # save_snapshot(batch, model, os.path.join(log_dir, "snapshot.png"),noise_pos=noise_pos)

    writer.close()
