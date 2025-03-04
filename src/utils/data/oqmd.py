"""Handle local dataset file built from Open Quantum Materials Database in CIF format."""
import os
import urllib.request

from .cif_dataset import CIFDataset
#from .data_base_classes import CIFDataset


class OQMD(CIFDataset):
    """
    Handle local dataset file built from Open Quantum Materials Database in CIF format.

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
        super().__init__(
            root,
            transform=transform,
            pre_filter=pre_filter,
            warn=warn,
            multithread=multithread,
            verbose=verbose,
        )

    @property
    def raw_file_names(self):
        return ["oqmd.cif"]

    @property
    def processed_file_names(self):
        return ["oqmd.hdf5"]

    def load(self):
        processed_file = os.path.join(self.processed_dir, self.processed_file_names[0])

        self.load_hdf5(processed_file)

    def download(self):
        pass

    def process(self):
        raw_file = os.path.join(self.raw_dir, self.raw_file_names[0])
        processed_file = os.path.join(self.processed_dir, self.processed_file_names[0])

        self.process_cif(
            raw_file,
            processed_file,
            loading_description=f"preprocess OQMD",
        )
