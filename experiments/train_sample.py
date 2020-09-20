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
        self.optimizer_w = optim.Adam(weight_network.parameters(), lr=opt.lr)

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

    def train(self):
        for epoch in range(self.n_epochs):
            for i, (batch_A, batch_B) in enumerate(zip(self.dataloader_A, self.dataloader_B)):

                real_A = batch_A[0].cuda().detach()
                real_B = batch_B[0].cuda().detach()
                labels_A = batch_A[1].cuda().detach()
                labels_B = batch_B[1].cuda().detach()

                # visualize_img_batch(real_A.cpu()) # Visualize batch
                # visualize_img_batch(real_B.cpu())
                
                # The weighting process
                w, unnorm_w = self.weight_network(real_A)
                
                sampled_idx_A = list( # Sample from batch A according to these importances
                    torch.utils.data.sampler.WeightedRandomSampler(w.squeeze(),
                                                                self.sampled_batch_size, 
                                                                replacement=True))
                w_sampled = w[sampled_idx_A]
                sampled_A = real_A[sampled_idx_A] # The sampled smaller batch A
                sampled_labs_A = labels_A[sampled_idx_A]

                # visualize_img_batch(sampled_A.cpu())
            
                # The loss function --------------------------------------------------------------------------------
                
                # Using f as objective function
                if self.opt.objective_function == 0:
                  L_A, L_B = f_0(sampled_labs_A * (w_sampled/w_sampled.detach()).view(-1), labels_B)
                else : 
                  L_A, L_B = self.objective_function(sampled_A, real_B, w_sampled)
                
                loss_w = ((L_A - L_B)**2).sum() # if f is a hidden variable, L_A and L_B are tensors, hence the sum() after the square
                
                self.mean_A += [real_A.mean()]
                self.mean_B += [real_B.mean()]

                # ---------------------------------------------------------------------------------------------------

                # Backward
                if self.is_train:
                    self.optimizer_w.zero_grad()
                    loss_w.backward()
                    self.optimizer_w.step()   

                # Store values --------------------------------------------------------------------------------------
                self.L_As += [L_A.sum().item()] # if f is a hidden variable, L_A and L_B are tensors, hence the sum()
                self.L_Bs += [L_B.sum().item()]
                
                self.losses_w += [loss_w.item()]
                
                w_a = self.weight_normalization((self.weight_network(self.dataset_A.example_imgs.float().unsqueeze(1).cuda())[0]))
                self.example_importances_A += [(w_a[0].item(), w_a[1].item())] # Store examples in a list

                # VALIDATION statistics: every once in a while during training, we compute the loss and weights on the validation (test) set
                if False: # i % 5 == 0: # compute avg and var every 5 steps because it's quite slow
                    '''
                     # compute mean and var for the weights and for unnormalized weights (on the training set)
                    mean, var, ratio01, unnorm_ratio01, unnorm_mean, unnorm_var = compute_average_prob(self.weight_network, self.dataloader_A, self.dataloader_B)
                    for key in mean.keys():
                        if (key not in self.w_means.keys()):
                            self.w_means[key] = []
                            self.w_vars[key] = [] 
                        self.w_means[key] += [unnorm_mean[key]] #[mean[key]]
                        self.w_vars[key] += [unnorm_var[key]] #[var[key]]
                        self.ratio01s += [ratio01]
                        self.unnorm_ratio01s += [unnorm_ratio01]'''
                    # compute mean and var for the weights and for unnormalized weights (on the test set)
                    mean, var, ratio01, unnorm_ratio01, unnorm_mean, unnorm_var = compute_average_prob(self.weight_network, self.testloader_A, self.testloader_B)
                    for key in mean.keys():
                        if (key not in self.w_means.keys()):
                            self.w_means[key] = []
                            self.w_vars[key] = [] 
                        self.w_means[key] += [unnorm_mean[key]] #[mean[key]]
                        self.w_vars[key] += [unnorm_var[key]] #[var[key]]
                        self.ratio01s += [ratio01]
                        self.unnorm_ratio01s += [unnorm_ratio01]
                    
                    # compute the loss on the test set
                    for (tbatch_A, tbatch_B) in zip(self.testloader_A, self.testloader_B):
                      treal_A = tbatch_A[0].cuda().detach()
                      treal_B = tbatch_B[0].cuda().detach()
                      tlabels_A = tbatch_A[1].cuda().detach()
                      tlabels_B = tbatch_B[1].cuda().detach()
                      
                      # The weighting process
                      tw, tunnorm_w = self.weight_network(real_A)
                      
                      tsampled_idx_A = list( # Sample from batch A according to these importances
                          torch.utils.data.sampler.WeightedRandomSampler(tw.squeeze(),
                                                                      self.sampled_batch_size, 
                                                                      replacement=True))
                      tw_sampled = tw[tsampled_idx_A]
                      tsampled_A = treal_A[tsampled_idx_A] # The sampled smaller batch A
                      tsampled_labs_A = tlabels_A[tsampled_idx_A]
                  
                      # The loss function --------------------------------------------------------------------------------
                      # Using f as objective function
                      if self.opt.objective_function == 0:
                        tL_A, tL_B = f_0(tlabels_A, tlabels_B, w)
                      else : 
                        tL_A, tL_B = self.objective_function(tsampled_A, treal_B, tw_sampled)
                      
                      tloss_w = ((tL_A - tL_B)**2).sum() # if f is a hidden variable, L_A and L_B are tensors, hence the sum() after the square
                    
                    self.test_losses_w += [tloss_w.item()]

                # ---------------------------------------------------------------------------------------------------

                # Print statistics
                if i % 5 == 0:
                    print('epoch', epoch, 'step', i, 'train_loss_w: ', loss_w.item())#, 'test_loss_w', tloss_w.item())
                    
                if i % self.opt.max_steps == 0 and i != 0:
                    break

        self.mean, self.var, self.ratio01, self.unnorm_ratio01, self.unnorm_mean, self.unnorm_var = compute_average_prob(self.weight_network, self.dataloader_A, self.dataloader_B)
