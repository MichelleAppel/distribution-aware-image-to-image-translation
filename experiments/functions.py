import torch

import collections

import matplotlib.pyplot as plt
import cv2


def visualize_img_batch(batch):
    '''Visualizes image batch
    
    Parameters:
    batch (Tensor): An image batch
    '''
    grid = make_grid(batch.unsqueeze(1).unsqueeze(1).cpu(), nrow=8, padding=1, normalize=False, range=None, scale_each=False, pad_value=0.5)
    plt.imshow(grid.permute(1,2,0))
    plt.show()

def visualize_MNIST_img_batch(batch):
    '''Visualizes image batch for MNIST
    
    Parameters:
    batch (Tensor): An image batch
    '''
    fig = plt.figure()
    for i in range(batch.shape[0]):
      plt.subplot(1,2,i+1)
      plt.tight_layout()
      plt.imshow(batch[i], cmap='gray', interpolation='none')
      plt.title("Ground Truth: {}".format(i))
      plt.xticks([])
      plt.yticks([])

def plot_hist(data):
    data = data.squeeze().cpu()
    plt.hist(data[data==1], weights=torch.ones(len(data[data==1]))/len(data), 
             color='black', bins=10, range= (0, 1))
    plt.hist(data[data==0], weights=torch.ones(len(data[data==0]))/len(data), 
             color='white', bins=10, range= (0, 1))
    plt.legend(['Imgs for label {}'.format(1), 
                'Imgs for label {}'.format(0)])
    plt.gca().set_facecolor('xkcd:gray')
    x_unique_count = torch.stack([(data==x_u).sum() for x_u in data.unique()])
    plt.show()


def compute_average_prob(weight_network, dataloader_A, dataloader_B):
    weights_total = collections.defaultdict(torch.tensor)
    weights_mean = collections.defaultdict(float)
    weights_var = collections.defaultdict(float)
    mean_weight_batch = collections.defaultdict(float)
    ratio01s = []
    unnorm_weights_batch_list = collections.defaultdict(torch.tensor)
    
    for i, (batch_A, batch_B) in enumerate(zip(dataloader_A, dataloader_B)):

        real_A = batch_A[0].cuda()
        labels_A = batch_A[1].cuda()
        weights_batch, unnorm_weights_batch = weight_network(real_A)

        possible_labels, _ = torch.unique(labels_A).sort()
        for l in possible_labels:
            indices_with_label_l = labels_A == l.item()
            if l not in weights_total.keys():
              weights_total[l.item()] = weights_batch[indices_with_label_l]
            else:
              torch.cat((weights_total[l.item()], weights_batch[indices_with_label_l]), 0)
            
            mean_weight_batch[l.item()] = weights_batch[indices_with_label_l].mean().item() # mean weight assigned to '0' and '1' for each batch: p('0',batch_i) and p('1',batch_i)

            if l not in weights_total.keys():
              unnorm_weights_batch_list[l.item()] = torch.exp(unnorm_weights_batch[indices_with_label_l])
            else:
              torch.cat((unnorm_weights_batch_list[l.item()], torch.exp(unnorm_weights_batch[indices_with_label_l])), 0) #all w_i for all i that are '0', lets call this list_0; the same for '1', lets call it list_1

        ratio01s += [mean_weight_batch[0] / mean_weight_batch[1]] # ratio q(batch_i) = p('0',batch_i)/p('1',batch_i)
        
    for key in weights_total.keys():
        weights_mean[key] = weights_total[key].mean().item()
        weights_var[key] = weights_total[key].var().item()
    
    ratio01 = torch.tensor(ratio01s).mean().item() # average over all q(batch_i)

    unnorm_ratio01 = torch.tensor(unnorm_weights_batch_list[0]).mean().item() / torch.tensor(unnorm_weights_batch_list[1]).mean().item() # average over list_0, and independently over list_1, and take the quotient

    return weights_mean, weights_var, ratio01, unnorm_ratio01
