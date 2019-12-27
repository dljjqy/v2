from torch.utils.data import Dataset
from pathlib import Path
from collections import namedtuple
from PIL import Image
import torch
from torchvision import transforms as T
# from utils import AUG

import numpy as np
Pair = namedtuple('Pair', ['image', 'mask'])

class BaseDataSet(Dataset):
    def __init__(self, root):
        """
        Initilize parameters
        root (Path): base root for original data and labeled data.
        augment (Bool): wheter does augmentation or not.
        """
        self.root = Path(root)
        self.files = []
        self._correspond()

    def _correspond(self):
        '''
        1.Set original data root and labeled data root. 
        2.Fill self.files with namedtuples each composed by the root of train original data and labeled data.
        
        file_path = namedtuple('file_path', ['original_path', 'labeled_path'])
        
        Tips: Must be called before _load_data
        '''
        raise NotImplementedError

    def _load_data(self, index):
        '''
        Cooperate with __getitem__,This method should be implemented based on the dataset's specilities.

        Aim: Return a pair contains a piece of original data and labeled data which readed from the file path.
        pair = namedtuple('pair', ['original', 'labeled'])
        '''
        raise NotImplementedError

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        '''
        Return a named tuple. pair = (origin = xx, label = xx)
        First: Fetch a pair of original data and labeled data.
        Second: Augment the pair of data in 'train' or 'val' way.
        '''
        if index >= self.__len__() or index < 0:
            raise IndexError
        else:
            pair = self._load_data(index)
        return pair