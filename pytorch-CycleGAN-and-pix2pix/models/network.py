import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class f_Net_2cl2fcl(nn.Module):
    ''' NN with 2 convolutional layers and fully connected layers'''

    def __init__(self):
        super(f_Net_2cl2fcl, self).__init__()
        self.softmax = nn.Softmax(dim=0)

        self.conv1 = nn.Conv2d(1, 4, kernel_size=5, stride = 2)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=5, stride = 2)
        self.fc1 = nn.Linear(128, 36)
        self.fc2 = nn.Linear(36, 1)

        # randomly initialize the network
        nn.init.uniform_(self.conv1.weight)
        nn.init.uniform_(self.conv2.weight)
        nn.init.uniform_(self.fc1.weight)
        nn.init.uniform_(self.fc2.weight)

        # freezes the network (does not update the weights in Grad Desc.)
        for param in self.parameters():
           param.requires_grad = False
        
    def forward(self, x):
        cl1 = torch.sigmoid(self.conv1(x))
        cl2 = torch.sigmoid(self.conv2(cl1))
        cl2 = cl2.view(-1, 128)
        fc1 = torch.sigmoid(self.fc1(cl2))
        fc2 = self.fc2(fc1)
        fc2 = fc2.squeeze()

        # outputs a dictionary
        out = {}
        out['cl1'] = cl1.view(-1, 576) # hidden variables
        out['cl2'] = cl2 # hidden variables
        out['fc1'] = fc1 # hidden variables
        out['out'] = fc2 # output
        
        return out

class f_Net_1cl1fcl(nn.Module):
    ''' NN with 1 convolutional layer and 1 fully connected layer'''

    def __init__(self):
        super(f_Net_1cl1fcl, self).__init__()

        self.conv1 = nn.Conv2d(1, 4, kernel_size=5, stride = 4)
        self.fc2 = nn.Linear(144, 1)

        # randomly initialize the network
        nn.init.uniform_(self.conv1.weight)
        nn.init.uniform_(self.fc2.weight)

        # freezes the network (does not update the weights in Grad Desc.)
        for param in self.parameters():
           param.requires_grad = False
        
    def forward(self, x):
        cl1 = torch.sigmoid(self.conv1(x))
        cl1 = cl1.view(-1, 144)
        fc2 = torch.sigmoid(self.fc2(cl1))
        fc2 = fc2.squeeze()

        # outputs a dictionary
        out = {}
        out['cl1'] = cl1 # hidden variables
        out['out'] = fc2 # output
        
        return out

class f_Net_4fcl(nn.Module):
    ''' NN with 4 fully connected layers'''

    def __init__(self):
        super(f_Net_4fcl, self).__init__()

        self.fc1 = nn.Linear(784, 392)
        self.fc2 = nn.Linear(392, 128)
        self.fc3 = nn.Linear(128, 36)
        self.fc4 = nn.Linear(36, 1)

        # randomly initialize the network
        nn.init.uniform_(self.fc1.weight)
        nn.init.uniform_(self.fc2.weight)
        nn.init.uniform_(self.fc3.weight)
        nn.init.uniform_(self.fc4.weight)

        # freezes the network (does not update the weights in Grad Desc.)
        for param in self.parameters():
           param.requires_grad = False
        
    def forward(self, x):
        x = x.view(-1, 784)
        fc1 = torch.sigmoid(self.fc1(x))
        fc2 = torch.sigmoid(self.fc2(fc1))
        fc3 = torch.sigmoid(self.fc3(fc2))
        fc4 = torch.sigmoid(self.fc4(fc3))
        fc4 = fc4.squeeze()

        # outputs a dictionary
        out = {}
        out['fc1'] = fc1 # hidden variables
        out['fc2'] = fc2 # hidden variables
        out['fc3'] = fc3 # hidden variables
        out['out'] = fc4 # output
        
        return out

class f_Net_3fcl(nn.Module):
    ''' NN with 3 fully connected layers'''

    def __init__(self):
        super(f_Net_3fcl, self).__init__()

        self.fc1 = nn.Linear(784, 392)
        self.fc2 = nn.Linear(392, 36)
        self.fc4 = nn.Linear(36, 1)

        # randomly initialize the network
        nn.init.uniform_(self.fc1.weight)
        nn.init.uniform_(self.fc2.weight)
        nn.init.uniform_(self.fc4.weight)

        # freezes the network (does not update the weights in Grad Desc.)
        for param in self.parameters():
           param.requires_grad = False
        
    def forward(self, x):
        x = x.view(-1, 784)
        fc1 = torch.sigmoid(self.fc1(x))
        fc2 = torch.sigmoid(self.fc2(fc1))
        fc3 = torch.sigmoid(self.fc4(fc2))
        fc3 = fc3.squeeze()

        # outputs a dictionary
        out = {}
        out['fc1'] = fc1 # hidden variables
        out['fc2'] = fc2 # hidden variables
        out['out'] = fc3 # output

        return out

class f_Net_2fcl(nn.Module):
    ''' NN with 2 fully connected layers'''

    def __init__(self):
        super(f_Net_2fcl, self).__init__()

        self.fc1 = nn.Linear(784, 36)
        self.fc4 = nn.Linear(36, 1)

        # randomly initialize the network
        nn.init.uniform_(self.fc1.weight)
        nn.init.uniform_(self.fc4.weight)

        # freezes the network (does not update the weights in Grad Desc.)
        for param in self.parameters():
           param.requires_grad = False
        
    def forward(self, x):
        x = x.view(-1, 784)
        fc1 = torch.sigmoid(self.fc1(x))
        fc2 = torch.sigmoid(self.fc4(fc1))
        fc2 = fc2.squeeze()

        # outputs a dictionary
        out = {}
        out['fc1'] = fc1 # hidden variables
        out['out'] = fc2 # output
        
        return out

class f_Net_1fcl(nn.Module):
    ''' NN with 1 fully connected layer'''

    def __init__(self):
        super(f_Net_1fcl, self).__init__()

        self.fc1 = nn.Linear(784, 1)

        # randomly initialize the network
        nn.init.uniform_(self.fc1.weight)

        # freezes the network (does not update the weights in Grad Desc.)
        for param in self.parameters():
           param.requires_grad = False
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.sigmoid(self.fc1(x))
        x = x.squeeze()
        
        # outputs a dictionary
        out = {}
        out['out'] = x # output
        
        return out


def f_0(labels_A, labels_B):
   '''f = label of the image '''

   L_A = (labels_A.float()).sum()
   L_B = (labels_B.float()).mean()

   return L_A, L_B

def f_1(real_A, real_B):
   '''f = mean of each image '''

   L_A = (torch.mean(real_A, dim=[2,3])).sum()
   L_B = (torch.mean(real_B, dim=[2,3])).sum()

   return L_A, L_B

def f_2(real_A, real_B):
   '''f = image itself; each image is weighted, and the sum across all images is computed '''

   #L_A  = (real_A * weight_normalization(w).repeat(28,28,1,1).permute(2,3,0,1)).sum()
   #L_B = (real_B).mean()
   
   L_A = (real_A).mean()
   L_B = (real_B).mean()

   return L_A, L_B

def f_3(real_A, real_B):
   '''f = output of a fixed (not-trained and randomly initialized) NN for the image'''
   
   ''' Architectures that work: '''
   f=f_Net_1fcl().cuda() # 1 fcl
   #f=f_Net_2fcl().cuda() # 2 fcl

   ''' Architectures that do not work: '''
   #f=f_Net_1cl1fcl().cuda() # 1 cl 1 fcl  
   #f=f_Net_3fcl().cuda() # 3 fcl
   #f=f_Net_4fcl().cuda() # 4 fcl
   #f=f_Net_2cl2fcl().cuda() # 2 cl 2 fcl
   
   L_A = (f(real_A)['out'].detach()).mean()
   L_B = (f(real_B)['out'].detach()).mean()

   return L_A, L_B

def weight_normalization(w):
    return w
    # return 0.5*(1 + w)

def f_4(real_A, real_B, w):
   '''f = hidden values of a fixed (not-trained and randomly initialized) NN for the image'''
   
   ''' Architectures that work: '''
   #f=f_Net_1fcl().cuda() # 1 fcl
   #f=f_Net_2fcl().cuda() # 2 fcl

   ''' Architectures that do not always work: '''
   f=f_Net_3fcl().cuda() # 3 fcl : works for f = fc1 or fc2, doesn't work for f = out
   #f=f_Net_4fcl().cuda() # 4 fcl : works for f = fc1 or fc2, doesn't work for f = fc3 or out
   #f=f_Net_1cl1fcl().cuda() # 1 cl 1 fcl : works for f = cl1, doesn't work for f = out   
   #f=f_Net_2cl2fcl().cuda() # 2 cl 2 fcl : works for f = cl1 or cl2, doesn't work for f = fc1 or out

   ''' Choose which part of the NN to use as f (output or any hidden variable) by selecting the correct entry in the dictionary output of the NN '''
   f_A = f(real_A)['fc1']
   if len(list(f_A.shape)) == 1:
     weights = weight_normalization(w).view(-1)
   else:
     dim = torch.cat([torch.tensor([1]), torch.tensor(f_A.shape[1:]).long()], dim = 0)
     weights = weight_normalization(w).repeat(tuple(dim))
   f_B = f(real_B)['fc1']
   
   L_A  = (f_A.detach() * weights).sum(dim = 0) # if f is a hidden variable, L_A and L_B are tensors, hence the sum(dim = 0)
   L_B = (f_B.detach()).mean(dim = 0)

   return L_A, L_B

class WeightNet(nn.Module):
    '''A simple network that predicts the importances of the samples'''

    def __init__(self):
        super(WeightNet, self).__init__()
#        self.fc1 = nn.Linear(1, 1)
#        self.fc2 = nn.Linear(1, 1)
        self.softmax = nn.Softmax(dim=0)
        self.sigmoid = nn.Sigmoid() # TODO: try relu

        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(2000, 40)
        self.fc2 = nn.Linear(40, 1)
        
    def forward(self, x):
        h1 = torch.sigmoid(F.max_pool2d(self.conv1(x), 2))
        h2 = torch.sigmoid(F.max_pool2d(self.conv2(h1), 2))
        h2_t = h2.view(-1, 2000)
        h3 = torch.sigmoid(self.fc1(h2_t))
        out = self.fc2(h3)
        return self.softmax(out), out