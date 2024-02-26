import os

from dataset.base_dataset import BaseDataset
from PIL import Image


class PairedDataset(BaseDataset):
    """Dataset of paired images from two domains.

    Args:
        data_dir (str): Directory containing 'train', 'val', and 'test' image folders.
        phase (str): One of 'train', 'val', or 'test'.
        resize_shape (tuple or list): Side lengths for resizing images.
        crop_shape (tuple or list): Side lengths for cropping images.
        direction (str): One of 'ab' or 'ba'.
    """
    def __init__(self, data_dir, phase, resize_shape, crop_shape, direction):
        self.str = "NOT USED"

    def __getitem__(self, index):

        return "not used"


    def __len__(self):
        return len(self.ab_paths)
