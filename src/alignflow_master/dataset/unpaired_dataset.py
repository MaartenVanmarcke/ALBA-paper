import os
import random

from dataset.base_dataset import BaseDataset
import numpy as np
import pathlib

class UnpairedDataset(BaseDataset):
    """Dataset of unpaired images from two domains.

    Args:
        data_dir (str): Directory containing 'train', 'val', and 'test' image folders.
        phase (str): One of 'train', 'val', or 'test'.
        shuffle_pairs (bool): Shuffle the pairs so that the image from domain B that appears
            with a given image from domain A is random.
        resize_shape (tuple or list): Side lengths for resizing images.
        crop_shape (tuple or list): Side lengths for cropping images.
        direction (str): One of 'ab' or 'ba'.
    """
    def __init__(self, data_dir, phase, shuffle_pairs, direction='ab'):
        if phase not in ('train', 'val', 'test'):
            raise ValueError('Invalid phase: {}'.format(phase))
        if direction not in ('ab', 'ba'):
            raise ValueError('Invalid direction: {}'.format(direction))

        if not isinstance(data_dir , dict):
            super(UnpairedDataset, self).__init__(data_dir, phase)
            current = pathlib.Path(__file__).parent.resolve().parent.resolve()
            self.a_file = os.path.join(current, "data", "testdata", "trainA.csv")
            self.a_data = np.genfromtxt(self.a_file, delimiter=',', dtype=np.float32)
            self.b_file = os.path.join(current, "data", "testdata", "trainB.csv")
            self.b_data = np.genfromtxt(self.b_file, delimiter=',', dtype=np.float32)
            self.data = {0: self.a_data, 1: self.b_data}
        else:
            self.data = data_dir
        '''if shuffle_pairs:
            np.random.shuffle(self.b_data)'''
        self.reverse = (direction == 'ba')
        self.shuffle_pairs = shuffle_pairs

    def __getitem__(self, index):
        
        '''return {'src': self.b_data[index,:] if self.reverse else self.a_data[index,:],
                'src_path': self.b_file if self.reverse else self.a_file,
                'tgt': self.a_data[index,:] if self.reverse else self.b_data[index,:],
                'tgt_path': self.a_file if self.reverse else self.b_file}'''
        
        res = {}
        for k, v in self.data.items():
            res[k] = v[index,:]
        return res

    def __len__(self):
        return min([d.shape[0] for d in self.data.values()])
