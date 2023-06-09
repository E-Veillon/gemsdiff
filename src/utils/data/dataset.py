import torch
from abc import ABCMeta, abstractmethod


class StructuresList(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def get_num_atoms(self) -> torch.LongTensor:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass
