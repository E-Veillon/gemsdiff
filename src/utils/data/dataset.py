"""Abstract dataset class to be subclassed for handling a sequence of structures data."""
import torch
from abc import ABCMeta, abstractmethod


class StructuresList(metaclass=ABCMeta):
    """Abstract dataset class to be subclassed for handling a sequence of structures data."""
    def __init__(self):
        pass

    @abstractmethod
    def get_num_atoms(self, idx: torch.LongTensor = None) -> torch.LongTensor:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass
