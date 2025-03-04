"""TODO: Module description."""
import torch
import numpy as np
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms
from ase.spacegroup import crystal
from PIL import Image
import tqdm
from tqdm.contrib.concurrent import process_map

import io
import typing as tp


def crystal_to_image_tensor(
    cell: torch.FloatTensor,
    x: torch.FloatTensor,
    z: torch.LongTensor,
    dpi: float|tp.Literal['figure'] = "figure",
    radii: float = 0.3,
) -> torch.ByteTensor:
    """
    TODO: function description.

    Parameters:
        cell (torch.FloatTensor):   Tensor concatenating structures unit cell parameters.

        x (torch.FloatTensor):      Tensor concatenating structures atomic positions.

        z (torch.LongTensor):       1-D Tensor of structures atomic numbers.

        dpi (float or 'figure'):    The resolution in dots per inch.
                                    If 'figure', use the figure's dpi value.
                                    Defaults to 'figure'.

        radii (float):              The radii of the atoms. Defaults to 0.3.

    Returns:
        torch.ByteTensor: TODO.
    """
    cry = crystal(
        z.cpu().numpy(),
        x.cpu().numpy(),
        cell=cell.cpu().numpy(),
    )

    plt.subplots_adjust(
        left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0, hspace=0.0
    )
    plot_atoms(cry, plt.gca(), radii=radii)
    plt.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, dpi=dpi, format="png")
    plt.close()
    buf.seek(0)
    im = Image.open(buf)
    tensor = torch.from_numpy(np.array(im))
    buf.close()

    tensor = torch.permute(tensor[:, :, :3], (2, 0, 1))

    return tensor


def fn_crystal_image(job):
    """
    TODO: function description.

    Parameters:
        job (TODO): TODO.

    Returns:
        tuple: TODO.
    """
    (i, t), rho, x, z = job
    tensor = crystal_to_image_tensor(rho, x, z)
    return ((i, t), tensor)


def make_video(
    rho: torch.FloatTensor,
    x: torch.FloatTensor,
    z: torch.LongTensor,
    num_atoms: torch.LongTensor,
    step: int = 1,
    max_workers: int = torch.get_num_threads(),
) -> torch.ByteTensor:
    """
    TODO: function description.

    Parameters:
        rho (torch.FloatTensor):        Tensor concatenating structures unit cell parameters.

        x (torch.FloatTensor):          Tensor concatenating structures atomic positions.

        z (torch.LongTensor):           1-D Tensor of structures atomic numbers.

        num_atoms (torch.LongTensor):   1-D Tensor of structures number of atoms.

        step (int):                     Time step between two shown frames.
                                        Defaults to 1 to get all frames shown.

        max_workers (int):              Number of parallel processes to spawn.
                                        Defaults to torch.get_num_threads().

    Returns:
        torch.ByteTensor: TODO.
    """
    n_struct = num_atoms.shape[0]
    idx = torch.arange(0, num_atoms.shape[0], step=step, device=num_atoms.device)

    batch = torch.arange(num_atoms.shape[0], device=num_atoms.device).repeat_interleave(
        num_atoms
    )

    I, T = torch.meshgrid(
        torch.arange(n_struct), torch.arange(idx.shape[0]), indexing="xy"
    )
    jobs = [
        ((i, t), rho[idx[t], i].cpu(), x[idx[t], batch == i].cpu(), z[batch == i].cpu())
        for i, t in zip(I.flatten(), T.flatten())
    ]

    results = process_map(
        fn_crystal_image,
        jobs,
        max_workers=max_workers,
        leave=False,
        desc="convertion video",
    )

    tensors = (
        torch.empty_like(results[0][1])
        .unsqueeze(0)
        .unsqueeze(1)
        .repeat(n_struct, idx.shape[0], 1, 1, 1)
    )
    for (i, t), tensor in results:
        tensors[i, t] = tensor

    return tensors
