import collections

import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from functions import compute_average_prob, visualize_img_batch
from network import f_0, f_1, f_2, f_3, f_4

class Train():

    def __init__(self, weight_network, dataset_A, dataloader_A, dataset_B, dataloader_B, opt, testloader_A, testloader_B, is_train=True):
        self.is_train = is_train
        self.opt = opt

        self.sampled_batch_size = opt.sampled_batch_size
        self.n_epochs = opt.n_epochs
        self.objective_function = [f_0, f_1, f_2, f_3, f_4][opt.objective_function]

        self.dataset_A = dataset_A
        self.dataset_B = dataset_B
        
        self.dataloader_A = dataloader_A
        self.dataloader_B = dataloader_B

        # Initialize the networks
        self.weight_network = weight_network

        # Optimizers
        self.optimizer_w = optim.Adam(self.weight_network.parameters(), lr=opt.lr)

        # Storing values
        self.losses_w = []

        self.mean_A = []
        self.mean_B = []

        self.L_As = []
        self.L_Bs = []

        self.example_importances_A = []
        self.example_importances_B = []

        self.w_means = collections.defaultdict(list)
        self.w_vars = collections.defaultdict(list)
        self.ratio01s = []
        self.unnorm_ratio01s = []

        self.mean = 0
        self.var = 0
        self.ratio01 = 0
        self.unnorm_ratio01 = 0

        # Test sets loaders
        self.testloader_A = testloader_A
        self.testloader_B = testloader_B
        self.test_losses_w = []

    def weight_normalization(self, w):
        return w
        # return 0.5*(1 + w)

    def label_norm(self, labels):
        '''so that label 0 has a signal'''
        return (labels - 0.5)*2

    def train(self):
        for epoch in range(self.n_epochs):
            for i, (batch_A, batch_B) in enumerate(zip(self.dataloader_A, self.dataloader_B)):
                real_A = batch_A[0].cuda()
                real_B = batch_B[0].cuda()
                labels_A = batch_A[1].cuda()
                labels_B = batch_B[1].cuda()
                
                # The weighting process
                w, unnorm_w = self.weight_network(real_A)

                if self.opt.importance_sampling == 1: # importance samplling
                    sampled_idx_A = list( # Sample from batch A according to these importances
                        torch.utils.data.sampler.WeightedRandomSampler(w.squeeze(),
                                                                    self.sampled_batch_size, 
                                                                    replacement=True))
                    w_sampled = w[sampled_idx_A]
                    real_A = real_A[sampled_idx_A] * (w_sampled/w_sampled.detach()).repeat(28,28,1,1).permute(2,3,0,1) # The sampled smaller batch A
                    labels_A = labels_A[sampled_idx_A] * (w_sampled/w_sampled.detach()).view(-1)
                else: # batch_weighting
                    # real_A = real_A * unnorm_w.repeat(28,28,1,1).permute(2,3,0,1)
                    labels_A = self.label_norm(labels_A.view(-1)) * self.weight_normalization(unnorm_w).view(-1)
                    labels_B = self.label_norm(labels_B)

                # Using f as objective function
                if self.opt.objective_function == 0:
                  L_A, L_B = f_0(labels_A, labels_B)
                else : 
                  L_A, L_B = self.objective_function(real_A, real_B)
                
                loss_w = ((L_A - L_B)**2).sum() # if f is a hidden variable, L_A and L_B are tensors, hence the sum() after the square
                
                self.mean_A += [real_A.mean()]
                self.mean_B += [real_B.mean()]

                # Backward
                if self.is_train:
                    self.optimizer_w.zero_grad()
                    loss_w.backward()
                    self.optimizer_w.step()       

                # Store values --------------------------------------------------------------------------------------
                self.L_As += [L_A.sum().item()] # if f is a hidden variable, L_A and L_B are tensors, hence the sum()
                self.L_Bs += [L_B.sum().item()]
                
                self.losses_w += [loss_w.item()]
                
                w_a = self.weight_normalization((self.weight_network(self.dataset_A.example_imgs.float().unsqueeze(1).cuda())[1])).detach()
                self.example_importances_A += [(w_a[0].item(), w_a[1].item())] # Store examples in a list

                if i % 5 == 0:
                    print('epoch', epoch, 'step', i, 'train_loss_w: ', loss_w.item())

        self.mean, self.var, self.ratio01, self.unnorm_ratio01, self.unnorm_mean, self.unnorm_var = compute_average_prob(self.weight_network, self.dataloader_A, self.dataloader_B)
