"""Write a CIF file from concatenated structures informations."""
import torch
from ase.spacegroup import crystal
from ase.io import write

from typing import List
import io


def make_cif(
    rho: torch.FloatTensor,
    x: torch.FloatTensor,
    z: torch.LongTensor,
    num_atoms: torch.LongTensor,
) -> List[str]:
    """
    Write a CIF file from concatenated structures informations.

    Parameters:
        rho (torch.FloatTensor):        Tensor concatenating structures unit cell parameters.

        x (torch.FloatTensor):          Tensor concatenating structures atomic positions.

        z (torch.FloatTensor):          1-D Tensor concatenating structures atomic numbers.

        num_atoms (torch.LongTensor):   1-D Tensor concatenating structures number of atoms.

    Returns:
        List[str]: List of the CIF formatted strings corresponding to the input structures.
    """
    n_struct = num_atoms.shape[0]

    batch = torch.arange(n_struct, device=rho.device).repeat_interleave(
        num_atoms
    )

    crystals = [
        crystal(z[batch == i].cpu(), x[batch == i].cpu(), cell=rho[i].cpu())
        for i in range(n_struct)
    ]

    buf = io.BytesIO()
    write(buf, crystals, format="cif")
    buf.seek(0)
    cif = buf.read().decode()
    buf.close()

    return cif
