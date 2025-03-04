"""Download and handle CDVAE team's curated dataset 'Carbon-24'."""
import os
import urllib.request

#from .data_base_classes import CSVDataset, DownloadProgressBar
from .csv_dataset import CSVDataset
from src.utils.download_bar import DownloadProgressBar

url_carbon24 = "https://raw.githubusercontent.com/txie-93/cdvae/main/data/carbon_24/"


class Carbon24(CSVDataset):
    """
    Download and handle CDVAE team's curated dataset 'Carbon-24'.
    
    Parameters:
        root (str):                         Root directory where the dataset should be saved.

        subset (str):                       Whether to download the 'train', 'val' or 'test'
                                            set of Carbon-24 dataset.

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

        warn (bool):                        Whether to issue warnings while processing the dataset.
                                            Defaults to False.

        multithread (bool):                 Whether to use parallel behavior to process data faster.
                                            Defaults to True.

        verbose (bool):                     Whether to print the number of loaded structures and show
                                            processing advancement as a progress bar. Defaults to True.
    """
    def __init__(
        self,
        root: str,
        subset: str,
        transform=None,
        pre_filter=None,
        warn: bool = False,
        multithread: bool = True,
        verbose: bool = True,
    ):
        assert subset in ("train", "val", "test")

        self.subset = subset

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
        return [f"{self.subset}.csv"]

    @property
    def processed_file_names(self):
        return [f"{self.subset}.hdf5"]

    def download(self):
        url = os.path.join(url_carbon24, self.raw_file_names[0])

        with DownloadProgressBar(
            unit="B",
            unit_scale=True,
            miniters=1,
            desc=f"downloading {self.raw_file_names[0]}",
        ) as t:
            urllib.request.urlretrieve(
                url,
                filename=os.path.join(self.raw_dir, self.raw_file_names[0]),
                reporthook=t.update_to,
            )

    def load(self):
        processed_file = os.path.join(self.processed_dir, self.processed_file_names[0])

        self.load_hdf5(processed_file)

    def process(self):
        raw_file = os.path.join(self.raw_dir, self.raw_file_names[0])
        processed_file = os.path.join(self.processed_dir, self.processed_file_names[0])

        self.process_csv(
            raw_file,
            processed_file,
            loading_description=f"preprocess {self.subset} set of Carbon-24",
        )
