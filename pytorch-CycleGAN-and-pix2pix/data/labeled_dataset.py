import os.path
import random

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset

def MNIST_data(distribution=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], train=True, dataset='MNIST'):
    # ratio: percentage of zeroes
    #returns (data, labels) for MNIST with only zeroes and ones, with the given ratio

    if dataset == 'MNIST':
      data = torchvision.datasets.MNIST('./files/', train=train, download=True,
                              transform=torchvision.transforms.Compose([
                                torchvision.transforms.Grayscale(3),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ]))
    else:
      if train:
        split = 'train'
      else:
        split = 'test'
      data = torchvision.datasets.SVHN('./files/', split=split, download=True,
                              transform=torchvision.transforms.Compose([
                                torchvision.transforms.Resize(28, interpolation=Image.NEAREST), # same size as MNIST
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ]))     



    if dataset == 'MNIST':
      targets = data.targets
    else:
      targets = torch.Tensor(data.labels)    

    unique_labels = torch.Tensor(distribution).nonzero()
    idxm = [targets==label for label in unique_labels] 
    idx = [np.where(idxm[label])[0] for label in unique_labels]

    tot_labels = [(targets==label).sum().item() for label in unique_labels]
    dim_res = [tot_labels[label] / distribution[label] for label in unique_labels]
    dim_res = int(min(dim_res)) / 2

    valid_idx = []
    valid_idx_labels = [[] for label in unique_labels]
    for label in unique_labels:
      number_samples = distribution[label] * dim_res
      valid_idx_labels[label] = idx[label][:int(number_samples)]
      valid_idx = valid_idx + valid_idx_labels[label].tolist()

    valid_idx = torch.sort(torch.tensor(valid_idx)).values

    shuffle = torch.randperm(len(valid_idx))
    valid_idx = torch.Tensor(valid_idx.float())[shuffle].long()

    if dataset == 'MNIST':
      data.targets = data.targets[valid_idx]
    else:
      data.targets = data.labels[valid_idx]
    data.data = data.data[valid_idx]

    return data 

class LabeledDataset(BaseDataset):

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        self.opt = opt

        self.dataset_A = MNIST_data(distribution=opt.distribution_A, train=opt.isTrain, dataset=opt.dataset_A)
        self.dataset_B = MNIST_data(distribution=opt.distribution_B, train=opt.isTrain, dataset=opt.dataset_B)

        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def example(self, domain='A'):
        '''
        Returns an example from each digit in the domain
        
        '''
        if domain == 'A':
          dataset = self.dataset_A
          distribution = self.opt.distribution_A
        else:
          dataset = self.dataset_B
          distribution = self.opt.distribution_B

        labels = dataset.targets

        
        unique_labels = torch.Tensor(distribution).nonzero()
        idx = [[labels==i][0].nonzero()[0].item for i in unique_labels]
        img = [dataset[i][0].unsqueeze(0) for i in idx]

        ex = torch.cat(img, 0)
           
        return ex

    def __getitem__(self, index):     
        index_A = index % len(self.dataset_A)
        index_B = random.randint(0, len(self.dataset_B) - 1) # randomize the index for domain B to avoid fixed pairs.

        A = self.dataset_A[index_A]
        B = self.dataset_B[index_B]
        
        return {'A': A[0], 'B': B[0], 'A_targets': A[1], 'B_targets': B[1], 'A_paths': 'None', 'B_paths': 'None'}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(len(self.dataset_A), len(self.dataset_B))

