"""A sampler class to handle and iterate through structure graphs batches."""
import torch
from torch.utils.data import Sampler, Subset

from typing import Iterator, List

from .dataset import StructuresList
#from .data_base_classes import StructuresList


class StructuresSampler(Sampler[List[int]]):
    """
    A sampler class to handle and iterate through structure graphs batches.
    
    Parameters:
        dataset (StructuresList):   The dataset to sample.

        max_atoms (int):            Max number of atoms in each structure to consider.

        shuffle (bool):             Whether to randomly shuffle data samples.
    """
    def __init__(
        self, dataset: StructuresList, max_atoms: int, shuffle: bool = False
    ) -> None:
        assert isinstance(dataset, StructuresList) or (
            isinstance(dataset, Subset) and isinstance(dataset.dataset, StructuresList)
        )
        assert isinstance(max_atoms, int)
        assert isinstance(shuffle, bool)

        self.dataset = dataset
        self.max_atoms = max_atoms
        self.shuffle = shuffle

        self._make_batch()

    def _make_batch(self):
        if isinstance(self.dataset, Subset):
            idx = torch.tensor(self.dataset.indices, dtype=torch.long)
            num_atoms = self.dataset.dataset.get_num_atoms(idx=idx)
        else:
            num_atoms = self.dataset.get_num_atoms()

        if self.shuffle:
            idx = torch.randperm(num_atoms.shape[0])
        else:
            idx = torch.arange(num_atoms.shape[0], dtype=torch.long)

        num_atoms = num_atoms[idx]

        cumsum = torch.cumsum(num_atoms, 0)

        self.batch = []
        while cumsum.shape[0] > 0:
            limit = (cumsum <= self.max_atoms).sum().item()
            self.batch.append(idx[:limit].tolist())
            cumsum -= cumsum[limit - 1].item()
            cumsum = cumsum[limit:]
            idx = idx[limit:]

    def __iter__(self) -> Iterator[List[int]]:
        sampler_iter = iter(self.batch)
        while True:
            try:
                yield next(sampler_iter)
            except StopIteration:
                break
        self._make_batch()

    def __len__(self) -> int:
        return len(self.batch)
