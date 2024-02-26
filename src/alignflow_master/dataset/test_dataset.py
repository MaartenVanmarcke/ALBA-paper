import os

from dataset.base_dataset import BaseDataset
import pathlib
import numpy as np

class TestDataset(BaseDataset):
    """Dataset of paired images from two domains.

    Args:
        data_dir (str): Directory containing 'train', 'val', and 'test' image folders.
        phase (str): One of 'train', 'val', or 'test'.
        resize_shape (tuple or list): Side lengths for resizing images.
        crop_shape (tuple or list): Side lengths for cropping images.
        direction (str): One of 'ab' or 'ba'.
    """
    def __init__(self, data_dir, phase, direction='ab'):
        if phase not in ('train', 'val', 'test'):
            raise ValueError('Invalid phase: {}'.format(phase))
        if direction not in ('ab', 'ba'):
            raise ValueError('Invalid direction: {}'.format(direction))

        super(TestDataset, self).__init__(data_dir, phase)
        current = pathlib.Path(__file__).parent.resolve().parent.resolve()
        self.a_file = os.path.join(current, "data", "testdata", "testA.csv")
        self.a_data = np.genfromtxt(self.a_file, delimiter=',', dtype=np.float32)
        self.b_file = os.path.join(current, "data", "testdata", "testB.csv")
        self.b_data = np.genfromtxt(self.b_file, delimiter=',', dtype=np.float32)
        self.reverse = (direction == 'ba')

    def __getitem__(self, index):
        
        return {'src': self.b_data[index,:] if self.reverse else self.a_data[index,:],
                'src_path': self.b_file if self.reverse else self.a_file,
                'tgt': self.a_data[index,:] if self.reverse else self.b_data[index,:],
                'tgt_path': self.a_file if self.reverse else self.b_file}

    def __len__(self):
        return min(self.a_data.shape[0], self.b_data.shape[0])
