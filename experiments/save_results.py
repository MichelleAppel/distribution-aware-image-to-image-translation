import os

from csv import writer

import torch
import matplotlib.pyplot as plt


class SaveResults():

    def __init__(self, destination, train, opt):
        self.destination = destination
        self.train = train
        self.opt = opt

    def plot_meansandvars(self):
        fig, axs = plt.subplots(2, 2, figsize=(10,10))
        fig.suptitle('Weights means and variances collected during the training')

        axs[0, 0].plot(self.train.w_means[0])
        axs[0, 0].set_title('Mean weight for 0s')
        axs[0, 1].plot(self.train.w_vars[0], 'tab:orange')
        axs[0, 1].set_title('Variance of weights for 0s')
        axs[1, 0].plot(self.train.w_means[1], 'tab:green')
        axs[1, 0].set_title('Mean weight for 1s')
        axs[1, 1].plot(self.train.w_vars[1], 'tab:red')
        axs[1, 1].set_title('Variance of weights for 1s')

        for ax in axs[:,0]:
            ax.set(xlabel='iterations / 5', ylabel='Mean')
        for ax in axs[:,1]:
            ax.set(xlabel='iterations / 5', ylabel='Variance')

        plt.figure(figsize=(10,6))
        plt.title('Ratio between the weights assigned to 0 and 1: average in each batch, compute the ratio, then average through all the batches')
        plt.xlabel('Training iterations')
        plt.ylabel('Ratio 0/1')
        # plt.yscale('symlog')
        plt.plot(self.train.ratio01s)
        plt.legend(['ratio01'])
        plt.savefig(os.path.join(self.destination, 'ratio01s.png'))
        plt.close()

        plt.figure(figsize=(10,6))
        plt.title('Ratio between the averages of not normalized weights for the classes')
        plt.xlabel('Training iterations')
        plt.ylabel('Unnormalized Ratio 0/1')
        # plt.yscale('symlog')
        plt.plot(self.train.unnorm_ratio01s)
        plt.legend(['Unnorm_ratio01'])
        plt.savefig(os.path.join(self.destination, 'ratio01s_unnorm.png'))

    def plot_w_loss(self):
        plt.figure(figsize=(10,6))
        plt.title('Losses over iterations')
        plt.xlabel('Training iterations')
        plt.ylabel('Loss')
        # plt.yscale('symlog')
        plt.plot(self.train.losses_w)
        plt.legend(['W'])
        plt.savefig(os.path.join(self.destination, 'w_losses.png'))

    def plot_L_loss(self):
        plt.figure(figsize=(10,6))
        plt.title('Losses over iterations')
        plt.xlabel('Training iterations')
        plt.ylabel('Loss')
        plt.plot(self.train.L_As)
        plt.plot(self.train.L_Bs)
        plt.legend(['L_A', 'L_B'])
        plt.savefig(os.path.join(self.destination, 'L_losses.png'))

    def plot_ratios(self):
        plt.figure(figsize=(10,6))
        plt.title('Assigned importances for the toy example images over the course of training')
        plt.plot(self.train.example_importances_A)
        plt.legend(['Img A with value {} (p={})'.format(0, self.opt.ratio_A), 
                    'Img A with value {} (p={})'.format(1, 1-self.opt.ratio_A)])
        plt.ylabel('Assigned importance')
        plt.xlabel('Training iterations')
        plt.savefig(os.path.join(self.destination, 'ratios.png'))

    def plot_importances(self):
        lambd = torch.linspace(0, 1, 64).repeat(28,28,1,1).permute(3,2,0,1)
        lin_comb = lambd * self.train.dataset_A.example_imgs[0] + (1-lambd) * self.train.dataset_A.example_imgs[1]

        weights, _ = self.train.weight_network(lin_comb.cuda())
        weights = weights.cpu().detach().numpy()
        plt.figure(figsize=(10,6))
        plt.title('Assigned importances for linear combination between images of 0 and 1 [lambda * 0 + (1-lambda * 1)]')
        plt.plot(torch.linspace(0, 1, 64), weights)
        plt.ylabel('Assigned importance')
        plt.xlabel('Lambda value')
        plt.savefig(os.path.join(self.destination, 'importances.png'))

    def plot_means(self):
        plt.figure(figsize=(10,6))
        plt.title('Losses over iterations')
        plt.xlabel('Training iterations')
        plt.ylabel('Mean')
        plt.plot(self.train.mean_A)
        plt.plot(self.train.mean_B)
        plt.legend(['mean_A', 'mean_B'])
        plt.savefig(os.path.join(self.destination, 'means.png'))

    def write_data(self, train):
        opt = self.opt 
        file_name = opt.CSV_name
        file_path = os.path.join(opt.results_dir, file_name)

        data = [opt.experiment_name, 
                opt.n_epochs, 
                opt.ratio_A, 
                1-opt.ratio_A, 
                opt.ratio_B, 
                1-opt.ratio_B, 
                opt.batch_size_A, 
                opt.sampled_batch_size,
                opt.objective_function,
                'yes',
                train.mean[0],
                train.var[0],
                train.mean[1],
                train.var[1],
                train.mean[0]/train.mean[1],
                train.ratio01,
                train.unnorm_ratio01,
                opt.ratio_B/opt.ratio_A,
                (1-opt.ratio_B)/(1-opt.ratio_A),
                (opt.ratio_B/opt.ratio_A) / ((1-opt.ratio_B)/(1-opt.ratio_A)),
                ((1-opt.ratio_B)/(1-opt.ratio_A)) / (opt.ratio_B/opt.ratio_A)]

        with open(file_path, 'a+', newline='') as write_obj:
            # Create a writer object from csv module
            csv_writer = writer(write_obj)
            # Add contents of list as last row in the csv file
            csv_writer.writerow(data)
