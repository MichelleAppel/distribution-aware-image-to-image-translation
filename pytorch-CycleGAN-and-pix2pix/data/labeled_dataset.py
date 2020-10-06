import os.path
import random

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset

def binary_data(ratio=0.5, train=True, dataset='MNIST'):
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
      idxm0 = data.targets==0
      idxm1 = data.targets==1 
    else:
      idxm0 = torch.Tensor(data.labels)==0
      idxm1 = torch.Tensor(data.labels)==1       
    dim = len(idxm0)  

    n0 = torch.sum(idxm0)
    n1 = torch.sum(idxm1)
    tot = n0 + n1

    if ratio < n0.item()/tot.item():
      if ratio == 1:
        size = n0
      else:
        size = int(n1/(1-ratio))
      idx0 = np.where(idxm0)[0]
      idx0 = idx0[:int(size*ratio)]
      idx1 = np.where(idxm1)[0]
      #idx0 = [True if i in indices else False for i in range(len(idx0))]
    else:
      if ratio == 1:
        size = n1
      else:
        size = int(n0/ratio)
      idx0 = np.where(idxm0)[0]
      idx1 = np.where(idxm1)[0]
      idx1 = idx1[:int(size*(1-ratio))]
      #idx1 = [True if i in indices else False for i in range(len(idx1))]

    idx = idx0.tolist() + idx1.tolist()
    idxm = torch.tensor( [True if i in idx else False for i in range(dim)] )

    #labels = MNIST.train_labels[idxm]
    #data = MNIST.train_data[idxm]

    if dataset == 'MNIST':
      data.targets = data.targets[idx]
    else:
      data.targets = data.labels[idx]
    data.data = data.data[idx]

    return data 

class LabeledDataset(BaseDataset):

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        self.dataset_A = binary_data(ratio=opt.ratio_A, train=opt.isTrain, dataset=opt.dataset_A)
        self.dataset_B = binary_data(ratio=opt.ratio_B, train=opt.isTrain, dataset=opt.dataset_B)

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
        else:
          dataset = self.dataset_B

        labels = dataset.targets
        idx0 = [labels==0][0].nonzero()[0].item()
        idx1 = [labels==1][0].nonzero()[0].item()
        img0 = dataset[idx0][0].unsqueeze(0)
        img1 = dataset[idx1][0].unsqueeze(0)
        ex = torch.cat((img0, img1), 0)
              
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

