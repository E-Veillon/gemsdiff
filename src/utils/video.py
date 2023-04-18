import torch

import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms
from ase.spacegroup import crystal


def make_video(
    rho: torch.FloatTensor,
    x: torch.FloatTensor,
    z: torch.LongTensor,
    num_atoms: torch.LongTensor,
    step: int = 1,
) -> torch.ByteTensor:
    idx = torch.arange(0, rho.shape[0], step=step)
