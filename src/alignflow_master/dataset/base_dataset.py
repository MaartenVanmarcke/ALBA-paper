import os
import torchvision.transforms as transforms
import torch.utils.data as data

from PIL import Image


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


class BaseDataset(data.Dataset):
    """Base dataset of images from two domains, subclassed by `PairedDataset`
    and `UnpairedDataset`.

    Args:
        data_dir (str): Directory containing 'train', 'val', and 'test' image folders.
        phase (str): One of 'train', 'val', or 'test'.
    """
    def __init__(self, data_dir, phase):
        if phase not in ('train', 'val', 'test'):
            raise ValueError('Invalid phase: {}'.format(phase))

        super(BaseDataset, self).__init__()
        self.data_dir = data_dir
        self.phase = phase

    def __getitem__(self, idx):
        raise NotImplementedError('Subclass of BaseDataset must implement __getitem__')

    def __len__(self):
        raise NotImplementedError('Subclass of BaseDataset must implement __len__')
