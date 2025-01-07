import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

from ase.formula import Formula
from ase import data
import tqdm

import os

from src.utils.scaler import LatticeScaler
from src.utils.data import MP, OQMD, StructuresSampler
from src.utils.hparams import Hparams
from src.model.gemsnet import GemsNetDiffusion
from src.utils.cif import make_cif


def get_dataloader(path: str, dataset: str, batch_size: int):
    assert dataset in ["mp", "oqmd"]

    dataset_path = os.path.join(path, dataset)
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

    loader_test = DataLoader(test_set, batch_size=64, num_workers=4)

    return loader_test


if __name__ == "__main__":
    import argparse

    from torch.utils.tensorboard import SummaryWriter

    parser = argparse.ArgumentParser(description="train denoising model")
    parser.add_argument("formula")
    parser.add_argument("--checkpoint", "-c")
    parser.add_argument("--output", "-o", default="sampling.cif")
    parser.add_argument("--device", "-d", default="cuda")
    parser.add_argument("--threads", "-t", type=int, default=8)

    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)

    # basic setup
    device = args.device

    hparams = Hparams()
    hparams.from_json(os.path.join(args.checkpoint, "hparams.json"))

    input_num_atoms = 0
    input_z = []
    for chem, n in Formula(args.formula).count().items():
        input_num_atoms += n
        input_z += [data.atomic_numbers[chem]] * n
    input_num_atoms = torch.tensor([input_num_atoms], dtype=torch.long)
    input_z = torch.tensor(input_z, dtype=torch.long)

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
        torch.load(
            os.path.join(args.checkpoint, "best.pt"), map_location=torch.device(device)
        ),
        strict=False,
    )
    model.eval()

    with torch.no_grad():

        input_z = input_z.to(device)
        input_num_atoms = input_num_atoms.to(device)

        pred_rho, pred_x = model.sampling(input_z, input_num_atoms, verbose=True)

        cif = make_cif(pred_rho, pred_x, input_z, input_num_atoms)

        with open(args.output, "w") as fp:
            fp.write(cif)
