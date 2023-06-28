import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

import tqdm

import os

from src.utils.scaler import LatticeScaler
from src.utils.data import MP20, OQMD, StructuresSampler
from src.utils.hparams import Hparams
from src.model.gemsnet import GemsNetDiffusion
from src.utils.cif import make_cif


def get_dataloader(path: str, dataset: str, batch_size: int):
    assert dataset in ["mp-20", "oqmd"]

    dataset_path = os.path.join(path, dataset)
    if dataset == "mp-20":
        test_set = MP20(dataset_path, "test")
    elif dataset == "oqmd":
        data = OQMD(dataset_path)
        gen = torch.Generator().manual_seed(42)
        _, _, test_set = random_split(data, [0.9, 0.05, 0.05], generator=gen)

    loader_test = DataLoader(test_set, batch_size=batch_size, num_workers=4)

    return loader_test


if __name__ == "__main__":
    import argparse

    from torch.utils.tensorboard import SummaryWriter

    parser = argparse.ArgumentParser(description="train denoising model")
    parser.add_argument("--checkpoint", "-c")
    parser.add_argument("--output", "-o", default="sampling.cif")
    parser.add_argument("--dataset", "-D", default="oqmd")
    parser.add_argument("--dataset-path", "-dp", default="./data")
    parser.add_argument("--device", "-d", default="cuda")
    parser.add_argument("--threads", "-t", type=int, default=8)

    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)

    # basic setup
    device = args.device

    hparams = Hparams()
    hparams.from_json(os.path.join(args.checkpoint, "hparams.json"))

    loader_test = get_dataloader(args.dataset_path, args.dataset, 512)

    scaler = LatticeScaler().to(device)

    model = GemsNetDiffusion(
        lattice_scaler=scaler,
        features=hparams.features,
        knn=hparams.knn,
        num_blocks=hparams.layers,
        diffusion_steps=hparams.diffusion_steps,
        x_betas=hparams.x_betas,
    ).to(device)

    model.load_state_dict(
        torch.load(os.path.join(args.checkpoint, "best.pt")), strict=False
    )
    model.eval()

    with torch.no_grad():
        rho, x, z, num_atoms = [], [], [], []
        for idx, batch in enumerate(tqdm.tqdm(loader_test)):
            batch = batch.to(device)

            pred_rho, pred_x = model.sampling(batch.z, batch.num_atoms, verbose=True)

            rho.append(pred_rho)
            x.append(pred_x)
            z.append(batch.z)
            num_atoms.append(batch.num_atoms)

            cat_rho, cat_x, cat_z, cat_num_atoms = (
                torch.cat(rho, dim=0),
                torch.cat(x, dim=0),
                torch.cat(z, dim=0),
                torch.cat(num_atoms, dim=0),
            )

            cif = make_cif(cat_rho, cat_x, cat_z, cat_num_atoms)

            with open(args.output, "w") as fp:
                fp.write(cif)
