"""Custom progress bar based on tqdm package for datasets downloads."""
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    """Custom progress bar based on tqdm package for datasets downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

