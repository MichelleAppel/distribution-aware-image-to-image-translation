import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torch

class ImportanceSamplingDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        self.weight_network_A = None
        self.weight_network_B = None
        
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
    
    def set_weight_networks(self, weight_network_A, weight_network_B):
        self.weight_network_A = weight_network_A
        self.weight_network_B = weight_network_B

    def accept_sample(self, img, weight_network):
        # Returns True if the image is accepted, False if rejected
        out = weight_network(img.unsqueeze(0)).detach()
        weight = torch.sigmoid(out)
        return bool(list(torch.utils.data.sampler.WeightedRandomSampler([1-weight, weight], 1))[0])

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        limit = 100 # Choose random sample if no sample is drawn after this amount of steps

        all_idx_A = torch.randperm(self.A_size)[:limit]
        for i in all_idx_A:
            A_path = self.A_paths[i]
            A_img = Image.open(A_path).convert('RGB')
            A = self.transform_A(A_img)
            accept = self.accept_sample(A, self.weight_network_A)
            if accept:
                break

        all_idx_B = torch.randperm(self.B_size)[:limit]
        for i in all_idx_B:
            B_path = self.B_paths[i]
            B_img = Image.open(B_path).convert('RGB')
            B = self.transform_B(B_img)
            accept = self.accept_sample(B, self.weight_network_B)
            if accept:
                break

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
