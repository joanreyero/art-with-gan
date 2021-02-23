import glob
import random
import os

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


# Inspired from https://github.com/aitorzip/PyTorch-CycleGAN/blob/master/datasets.py

class ImageDataset(Dataset):
    # TODO Change mode to train
    def __init__(self, root, transform=None, mode=''):
        self.transform = transform
        
        self.files_A = sorted(glob.glob(os.path.join(root, '%smonet_jpg' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%sphoto_jpg' % mode) + '/*.*'))
        
        if len(self.files_A) > len(self.files_B):
            self.files_A, self.files_B = self.files_B, self.files_A
        self.new_perm()

    def new_perm(self):
        self.randperm = torch.randperm(len(self.files_B))[:len(self.files_A)]

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        item_B = self.transform(Image.open(self.files_B[self.randperm[index]]))
        if item_A.shape[0] != 3: 
            item_A = item_A.repeat(3, 1, 1)
        if item_B.shape[0] != 3: 
            item_B = item_B.repeat(3, 1, 1)
        if index == len(self) - 1:
            self.new_perm()
        # Old versions of PyTorch didn't support normalization for different-channeled images
        return (item_A - 0.5) * 2, (item_B - 0.5) * 2

    def __len__(self):
        return min(len(self.files_A), len(self.files_B))


if __name__ == '__main__':
    ds = ImageDataset('../data/')
    print(ds.files_B)
