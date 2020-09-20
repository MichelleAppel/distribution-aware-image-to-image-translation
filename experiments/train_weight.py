import torch
import torch.optim as optim
import collections

from functions import compute_average_prob
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

                # The weighting process
                w, unnorm_w = self.weight_network(real_A)
            
                # The loss function --------------------------------------------------------------------------------
                
                # Using f as objective function
                if self.opt.objective_function == 0:
                  L_A, L_B = f_0(labels_A, labels_B)
                else : 
                  L_A, L_B = self.objective_function(real_A, real_B, w)
                
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
                if i % 1 == 0: # compute avg and var every 5 steps because it's quite slow
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
                    for j, (batch_A_test, batch_B_test) in enumerate(zip(self.testloader_A, self.testloader_B)):
                      real_A_test = batch_A_test[0].cuda().detach()
                      real_B_test = batch_B_test[0].cuda().detach()
                      labels_A_test = batch_A_test[1].cuda().detach()
                      labels_B_test = batch_B_test[1].cuda().detach()
                      
                      # The weighting process
                      test_w, test_unnorm_w = self.weight_network(real_A_test)
                      
                      # The loss function --------------------------------------------------------------------------------
                      # Using f as objective function
                      if self.opt.objective_function == 0:
                        L_A_test, L_B_test = f_0(labels_A_test, labels_B_test, test_w)
                      else : 
                        L_A_test, L_B_test = self.objective_function(real_A_test, real_B_test, test_w)
                      
                      test_loss_w = ((L_A_test - L_B_test)**2).sum() # if f is a hidden variable, L_A and L_B are tensors, hence the sum() after the square
                    
                    self.test_losses_w += [test_loss_w.item()]

                # ---------------------------------------------------------------------------------------------------

                # Print statistics
                if i % 1 == 0:
                    print('epoch', epoch, 'step', i, 'train_loss_w: ', loss_w.item(), 'test_loss_w:', test_loss_w.item())
                    
                if i % self.opt.max_steps == 0 and i != 0:
                    break

        self.mean, self.var, self.ratio01, self.unnorm_ratio01, self.unnorm_mean, self.unnorm_var = compute_average_prob(self.weight_network, self.dataloader_A, self.dataloader_B)