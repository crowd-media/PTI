import os

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from configs import global_config
from utils.data_utils import make_dataset


class ImagesDataset(Dataset):

    def __init__(self, source_root, source_transform=None):
        self.source_paths = sorted(make_dataset(source_root))
        self.source_transform = source_transform

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        fname, from_path = self.source_paths[index]
        from_im = Image.open(from_path).convert('RGB')
         
        from_im = np.array(from_im)

        # convert image to tensor
        from_im = torch.tensor(from_im).permute(2,0,1).float().div(255.0).to(global_config.device)

        # normalize image
        from_im = (from_im - 0.5) / 0.5

        return fname, from_im
