import os
import urllib.request

from .cif_dataset import CIFDataset


class MP(CIFDataset):
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
        return ["mp.cif"]

    @property
    def processed_file_names(self):
        return ["mp.hdf5"]

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
            loading_description=f"proprocess MP",
        )
