import torch
from torch.utils.data import Dataset, DataLoader
import torchvision

import numpy as np

def binary_data(ratio=0.5, train=True, dataset='MNIST'):
    # ratio: percentage of zeroes
    #returns (data, labels) for MNIST with only zeroes and ones, with the given ratio

    if dataset == 'MNIST':
      data = torchvision.datasets.MNIST('./files/', train=train, download=True,
                              transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                  (0.1307,), (0.3081,))
                              ]))
    else:
      if train:
        split = 'train'
      else:
        split = 'test'
      data = torchvision.datasets.SVHN('./files/', split=split, download=True,
                              transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                  (0.1307,), (0.3081,))
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

class CustomDataset(Dataset):
    '''The dataset for the MNIST binary data
    '''
    def __init__(self, ratio=0.5, train=True, dataset='MNIST'):

        self.ratio = ratio
        
        self.dataset = binary_data(ratio=self.ratio, train=train, dataset=dataset)
        
        self.example_imgs = self.example()
        
    def example(self):
        '''
        Returns an example from each digit in the domain
        
        '''
        labels = self.dataset.targets
        data = self.dataset.data
        img0 = self.__getitem__([labels==0][0].nonzero()[0].item())
        img1 = self.__getitem__([labels==1][0].nonzero()[0].item())
        ex = torch.cat((img0, img1), 0)
              
        return ex

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):      
        return self.dataset[idx]