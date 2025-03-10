"""Abstract dataset class to be subclassed for handling of Crystallographic Information File (CIF) formatted datasets."""
from typing import Iterator, Sequence, Any
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader

import h5py
import torch
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase import Atoms
from ase.io import iread
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
import multiprocessing as mp
import warnings
import os
import json

from .dataset import StructuresList


def _process_atoms(atoms: Atoms) -> dict[str, Any]:
    cell = atoms.cell.array.astype(np.float32)
    z = np.array(atoms.get_atomic_numbers(), dtype=np.int64)
    pos = atoms.get_scaled_positions().astype(np.float32)
    [a, b, c, alpha, beta, gamma] = atoms.cell.cellpar()
    lengths = np.array([a, b, c], dtype=np.float32)
    angles = np.array([alpha, beta, gamma], dtype=np.float32)

    data = {"cell": cell, "lengths": lengths, "angles": angles, "z": z, "pos": pos}

    return data


class CIFDataset(InMemoryDataset, StructuresList, metaclass=ABCMeta):
    """
    Abstract dataset class to be subclassed for handling of Crystallographic Information File (CIF) formatted datasets.
    
    Parameters:
        root (str):                         Root directory where the dataset should be saved.

        transform (callable, optional):     A function/transform that takes in a
                                            :class:`~torch_geometric.data.Data` or
                                            :class:`~torch_geometric.data.HeteroData` object
                                            and returns a transformed version.
                                            The data object will be transformed before every access.
                                            (default: :obj:`None`)

        pre_filter (callable, optional):    A function that takes in a
                                            :class:`~torch_geometric.data.Data` or
                                            :class:`~torch_geometric.data.HeteroData` object
                                            and returns a boolean value, indicating whether the data
                                            object should be included in the final dataset.
                                            (default: :obj:`None`)

        warn (bool):                        TODO: unused argument ? (only used in CSVDataset ABC).
                                            Defaults to False.

        multithread (bool):                 Whether to use parallel behavior to process data faster.
                                            Defaults to True.

        verbose (bool):                     Whether to print the number of loaded structures and show
                                            processing advancement as a progress bar. Defaults to True.
    """
    def __init__(
        self,
        root: str,
        transform=None,
        pre_filter=None,
        warn: bool = False,
        multithread: bool = True,
        verbose: bool = True,
    ):
        """
        Abstract dataset class to be subclassed for handling of Crystallographic Information File (CIF) formatted datasets.

        Parameters:
            root (str):                         Root directory where the dataset should be saved.

            transform (callable, optional):     A function/transform that takes in a
                                                :class:`~torch_geometric.data.Data` or
                                                :class:`~torch_geometric.data.HeteroData` object
                                                and returns a transformed version.
                                                The data object will be transformed before every access.
                                                (default: :obj:`None`)

            pre_filter (callable, optional):    A function that takes in a
                                                :class:`~torch_geometric.data.Data` or
                                                :class:`~torch_geometric.data.HeteroData` object
                                                and returns a boolean value, indicating whether the data
                                                object should be included in the final dataset.
                                                (default: :obj:`None`)

            warn (bool):                        TODO: unused argument ? (only used in CSVDataset ABC).
                                                Defaults to False.

            multithread (bool):                 Whether to use parallel behavior to process data faster.
                                                Defaults to True.

            verbose (bool):                     Whether to print the number of loaded structures and show
                                                processing advancement as a progress bar. Defaults to True.
        """
        self.warn = warn
        self.multithread = multithread
        self.verbose = verbose

        super().__init__(root, transform, pre_filter=pre_filter)

        self.load()

        if self.verbose:
            print(f"{len(self)} structures loaded!")

    @abstractmethod
    def download(self):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def process(self):
        pass

    def get_num_atoms(self, idx: torch.LongTensor = None) -> torch.LongTensor:
        if idx is None:
            return torch.from_numpy(self.num_atoms)
        else:
            return torch.from_numpy(self.num_atoms[idx.numpy()])

    def load_hdf5(self, hdf5_file: str) -> None:
        f = h5py.File(hdf5_file, "r")

        self.material_id = f["material_id"][:]
        self.batch = f["batch"][:]
        self.num_atoms = f["num_atoms"][:]
        self.ptr = f["ptr"][:]
        self.cell = f["cell"][:]
        self.lengths = f["lengths"][:]
        self.angles = f["angles"][:]
        self.pos = f["pos"][:]
        self.z = f["z"][:]

        f.close()

        # basic tensor shape checking
        assert self.material_id.ndim == 1 and self.material_id.dtype == np.int32
        assert self.batch.ndim == 1 and self.batch.dtype == np.int64
        assert self.num_atoms.ndim == 1 and self.num_atoms.dtype == np.int64
        assert self.ptr.ndim == 1 and self.ptr.dtype == np.int64

        assert self.cell.ndim == 3 and self.cell.dtype == np.float32
        assert self.lengths.ndim == 2 and self.lengths.dtype == np.float32
        assert self.angles.ndim == 2 and self.angles.dtype == np.float32

        assert self.pos.ndim == 2 and self.pos.dtype == np.float32
        assert self.z.ndim == 1 and self.z.dtype == np.int64

        # checking size
        n_struct = self.num_atoms.shape[0]
        n_atoms = np.sum(self.num_atoms)

        assert self.material_id.shape == (n_struct,)
        assert self.batch.shape == (n_atoms,)
        assert self.ptr.shape == (n_struct + 1,) and self.ptr[-1] == n_atoms
        assert self.cell.shape == (n_struct, 3, 3)
        assert self.lengths.shape == (n_struct, 3)
        assert self.angles.shape == (n_struct, 3)
        assert self.pos.shape == (n_atoms, 3)
        assert self.z.shape == (n_atoms,)

        self.idx_filtered = torch.arange(self.num_atoms.shape[0], dtype=torch.long)
        if self.pre_filter is not None:
            mask = torch.tensor(
                [
                    self.pre_filter(self.get(idx))
                    for idx in range(self.num_atoms.shape[0])
                ],
                dtype=torch.bool,
            )
            self.idx_filtered = torch.arange(self.num_atoms.shape[0], dtype=torch.long)[
                mask
            ]

    def process_cif(
        self,
        cif_file: str,
        hdf5_file: str,
        loading_description: str = "loading dataset",
    ) -> None:
        iterator = iread(cif_file)

        if self.multithread:
            if self.verbose:
                results = process_map(
                    _process_atoms,
                    iterator,
                    desc=loading_description,
                    chunksize=8,
                )
            else:
                with mp.Pool(mp.cpu_count()) as p:
                    results = p.map(_process_atoms, iterator)
        else:
            results = []

            if self.verbose:
                iterator = tqdm(iterator, desc=loading_description)

            for args in iterator:
                results.append(_process_atoms(args))

        material_id = np.arange(len(results), dtype=np.int32)

        cell = np.stack([struct["cell"] for struct in results], axis=0).astype(
            np.float32
        )
        lengths = np.stack([struct["lengths"] for struct in results], axis=0).astype(
            np.float32
        )
        angles = np.stack([struct["angles"] for struct in results], axis=0).astype(
            np.float32
        )

        batch = np.concatenate(
            [
                np.full_like(struct["z"], fill_value=idx, dtype=np.int64)
                for idx, struct in enumerate(results)
            ],
            axis=0,
        )
        num_atoms = np.array(
            [struct["z"].shape[0] for struct in results], dtype=np.int64
        )
        z = np.concatenate([struct["z"] for struct in results], axis=0).astype(np.int64)
        pos = np.concatenate([struct["pos"] for struct in results], axis=0).astype(
            np.float32
        )

        ptr = np.pad(np.cumsum(num_atoms, axis=0), (1, 0))

        print(f"saving to {hdf5_file}")
        f = h5py.File(hdf5_file, "w")
        f.create_dataset("material_id", material_id.shape, dtype=material_id.dtype)[
            :
        ] = material_id
        f.create_dataset("batch", batch.shape, dtype=batch.dtype)[:] = batch
        f.create_dataset("num_atoms", num_atoms.shape, dtype=num_atoms.dtype)[
            :
        ] = num_atoms
        f.create_dataset("ptr", ptr.shape, dtype=ptr.dtype)[:] = ptr

        f.create_dataset("cell", cell.shape, dtype=cell.dtype)[:, :, :] = cell
        f.create_dataset("lengths", lengths.shape, dtype=lengths.dtype)[:, :] = lengths
        f.create_dataset("angles", angles.shape, dtype=angles.dtype)[:, :] = angles

        f.create_dataset("pos", pos.shape, dtype=pos.dtype)[:] = pos
        f.create_dataset("z", z.shape, dtype=z.dtype)[:] = z

        f.close()

    def len(self) -> int:
        return self.idx_filtered.shape[0]

    def get(self, idx: int) -> Data:
        idx = self.idx_filtered[idx]

        material_id = torch.tensor(self.material_id[idx])
        num_atoms = torch.tensor(self.num_atoms[idx])
        cell = torch.from_numpy(self.cell[idx]).unsqueeze(0)
        z = torch.from_numpy(
            self.z[self.ptr[idx] : self.ptr[idx] + self.num_atoms[idx]]
        )
        pos = torch.from_numpy(
            self.pos[self.ptr[idx] : self.ptr[idx] + self.num_atoms[idx]]
        )

        return Data(
            material_id=material_id,
            z=z,
            pos=pos,
            cell=cell,
            num_atoms=num_atoms,
        )
