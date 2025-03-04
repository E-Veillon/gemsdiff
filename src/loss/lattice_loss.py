"""Loss function comparing normalized lattice parameters (a, b, c, alpha, beta, gamma) distances between predicted and targeted lattices."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.scaler import LatticeScaler


class LatticeParametersLoss(nn.Module):
    """
    Loss function comparing normalized lattice parameters (a, b, c, alpha, beta, gamma)
    distances between predicted and targeted lattices.
    
    Parameters:
        lattice_scaler (LatticeScaler): An instance of the LatticeScaler class,
                                        used to manipulate lattices representation.

        distance (str):                 distance formula to use for comparison.
                                        Supports "l1" and "mse" distances.

    Returns:
        torch.Tensor: Tensor containing loss values.
    """
    def __init__(self, lattice_scaler: LatticeScaler = None, distance: str = "l1"):
        """Init."""
        super().__init__()
        assert distance in ["l1", "mse"]

        if lattice_scaler is None:
            lattice_scaler = LatticeScaler()

        assert isinstance(lattice_scaler, LatticeScaler)

        self.lattice_scaler = lattice_scaler
        self.distance = distance

    def forward(
        self,
        source: torch.FloatTensor | tuple[torch.FloatTensor, torch.FloatTensor],
        target: torch.FloatTensor | tuple[torch.FloatTensor, torch.FloatTensor],
    ) -> torch.FloatTensor:
        """
        Compute lattice parameters loss values for each pair of source and target unit cells.

        Parameters:
            source (torch.FloatTensor|(torch.FloatTensor, torch.FloatTensor)):  Source unit cells.

            target (torch.FloatTensor|(torch.FloatTensor, torch.FloatTensor)):  Target unit cells.

        Returns:
            torch.FloatTensor: 1-D tensor of the MAE or MSE loss values between each pair of cell.
        """
        if isinstance(target, tuple):
            param_src = self.lattice_scaler.normalise(source)
        else:
            param_src = self.lattice_scaler.normalise_lattice(source)

        if isinstance(target, tuple):
            param_tgt = self.lattice_scaler.normalise(target)
        else:
            param_tgt = self.lattice_scaler.normalise_lattice(target)

        y_src = torch.cat(param_src, dim=1)
        y_tgt = torch.cat(param_tgt, dim=1)

        if self.distance == "l1":
            return F.l1_loss(y_src, y_tgt)
        if self.distance == "mse":
            return F.mse_loss(y_src, y_tgt)

        raise Exception(f"unkown distance {self.distance}")
