from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import datasets, transforms
import os
import pandas as pd
import numpy as np
import cv2

is_cuda = torch.cuda.is_available()
kwargs = {'num_workers': 1, 'pin_memory': True} if is_cuda else {}

def train_loader(data, data_directory = '/home/sungwonlyu/data', batch_size = 128):
    if data == 'mnist':
        train_dataloader = DataLoader(
            datasets.MNIST(data_directory + '/' + data, train=True, download=True, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True, **kwargs)
    elif data == 'svhn':
        train_dataloader = DataLoader(
            datasets.SVHN(data_directory + '/' + data, train=True, download=True, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True, **kwargs)
    elif data == 'cifar10':
        train_dataloader = DataLoader(
            datasets.CIFAR10(data_directory + '/' + data, train=True, download=True, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True, **kwargs)
    elif data == 'alphachu':
        train_dataloader = DataLoader(
            AlphachuDataset(data_directory + '/' + data, train=True, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True, **kwargs)
    return train_dataloader

def test_loader(data, data_directory = '/home/sungwonlyu/data', batch_size = 128):
    if data == 'mnist':
        test_dataloader = DataLoader(
            datasets.MNIST(data_directory + '/' + data, train=False, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True, **kwargs)
    elif data == 'svhn':
        test_dataloader = DataLoader(
            datasets.SVHN(data_directory + '/' + data, train=False, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True, **kwargs)
    elif data == 'cifar10':
        test_dataloader = DataLoader(
            datasets.CIFAR10(data_directory + '/' + data, train=False, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True, **kwargs)
    elif data == 'alphachu':
        test_dataloader = DataLoader(
            AlphachuDataset(data_directory + '/' + data, train=False, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True, **kwargs)
    return test_dataloader


class AlphachuDataset(Dataset):
    """Alphachu dataset."""
    def __init__(self, root_dir, train = True, transform=None):
        """
            Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.makelist()

    def makelist(self):
        img_list = os.listdir(self.root_dir)
        test = len(img_list) // 10
        if not self.train:
            img_list = img_list[:test]
        else:
            img_list = img_list[test:]
        # timestamps = [int(i[:12]) for i in img_list]
        # sets = [int(i[12:].split('-')[0]) for i in img_list]
        frames = [int(i.split('-')[1].split('.')[0]) for i in img_list]
        img_list = pd.DataFrame(img_list)
        # img_list['timestamps'] = timestamps
        # img_list['sets'] = sets
        img_list['frames'] = frames
        self.img_list = img_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.root_dir + '/' + self.img_list.iloc[idx, 0]
        image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        image = np.expand_dims(np.array(image), 2)
        if self.transform:
            image = self.transform(image)
        frames = self.img_list.iloc[idx, 1]
        # timestamps = self.img_list.iloc[idx, 1]
        # sets = self.img_list.iloc[idx, 2]
        # sample = {'image': image, 'set': sets, 'frame': frames}
        return image, frames
