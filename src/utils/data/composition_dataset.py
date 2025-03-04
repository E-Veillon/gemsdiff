"""A dataset class to load and handle chemical compositions from given JSON file."""
from collections.abc import Sequence
from typing import Iterator
from torch_geometric.data import InMemoryDataset, Data

import torch
import torch.nn.functional as F
import numpy as np

import os
import json

from .dataset import StructuresList
from ase.data import chemical_symbols, atomic_names

symboles = {s: z for z, s in enumerate(chemical_symbols)}
names = {n: z for z, n in enumerate(atomic_names)}
str_to_z = symboles | names


class CompositionDataset(InMemoryDataset, StructuresList):
    """
    A dataset class to load and handle chemical compositions represented as lists
    of constitutive elements symbols or atomic numbers from a JSON file.

    Parameters:
        file (str): Path to the JSON file to load compositions from.
    """
    def __init__(self, file: str):
        """
        A dataset class to load and handle chemical compositions represented as lists
        of constitutive elements symbols or atomic numbers from a JSON file.

        Parameters:
            file (str): Path to the JSON file to load compositions from.
        """
        self.file = file

        self.transform = None
        self.pre_transform = None
        self.pre_filter = None

        self.load()

    def download(self):
        pass

    def load(self):
        with open(self.file, "r") as fp:
            compositions = json.load(fp)

        self.num_atoms = torch.tensor(
            [len(comp) for comp in compositions], dtype=torch.long
        )
        self.ptr = F.pad(torch.cumsum(self.num_atoms, 0), (1, 0), value=0)

        list_comp = []
        for comp in compositions:
            elems = []
            for elem in comp:
                if isinstance(elem, str):
                    z = str_to_z.get(elem, None)
                    assert z is not None, f"{elem} value is unknown"
                elif isinstance(elem, int):
                    z = elem
                else:
                    raise Exception(f"{elem} value is unknown")
                elems.append(z)
            list_comp.append(elems)
        self.z = torch.tensor(sum(list_comp, []))

    def process(self):
        pass

    def get_num_atoms(self) -> torch.LongTensor:
        return self.num_atoms

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [self.file]

    def len(self) -> int:
        return self.idx_filtered.shape[0]

    def indices(self) -> Sequence:
        return range(self.num_atoms.shape[0])

    def get(self, idx: int) -> Data:
        num_atoms = self.num_atoms[idx]
        z = self.z[self.ptr[idx] : self.ptr[idx] + self.num_atoms[idx]]

        return Data(z=z, num_atoms=num_atoms)
