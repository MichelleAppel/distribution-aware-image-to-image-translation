"""Model class template

This module provides a template for users to implement custom models.
You can specify '--model template' to use this model.
The class name should be consistent with both the filename and its model option.
The filename should be <model>_dataset.py
The class name should be <Model>Dataset.py
It implements a simple image-to-image translation baseline based on regression loss.
Given input-output pairs (data_A, data_B), it learns a network netG that can minimize the following L1 loss:
    min_<netG> ||netG(data_A) - data_B||_1
You need to implement the following functions:
    <modify_commandline_options>:　Add model-specific options and rewrite default values for existing options.
    <__init__>: Initialize this model class.
    <set_input>: Unpack input data and perform data pre-processing.
    <forward>: Run forward pass. This will be called by both <optimize_parameters> and <test>.
    <optimize_parameters>: Update network weights; it will be called in every training iteration.
"""
import torch
from .base_model import BaseModel
from . import networks

import torch.nn as nn
from util.image_pool import ImagePool
import itertools


class BatchWeightModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        parser.set_defaults(dataset_mode='unaligned') # Unaligned
        
        # The network architectures of D and G
        parser.set_defaults(netG='batch_weight')
        parser.set_defaults(netD='joint')

        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        # specify the training losses you want to print out. The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        self.loss_names = ['G', 'W', 'D', 'minus', 'plus']
        
        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        visual_names_A = ['real_A', 'fake_B', 'rec_A', 'idt_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B', 'idt_B']
        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        
        # The models to load
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D', 'W_A', 'W_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B'] # No W's during test time
        
        # define networks
        # generators
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        
        if self.isTrain:
            # discriminator
            self.netD = networks.define_D(opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, opt.output_nc)

            # weight networks
            self.netW_A = networks.define_W(gpu_ids=self.gpu_ids, ngf=opt.ngf)
            self.netW_B = networks.define_W(gpu_ids=self.gpu_ids, ngf=opt.ngf)

            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images

            # define loss
            self.criterionGAN = networks.weighted_GANLoss().to(self.device)

            # define and initialize optimizers. 
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_W = torch.optim.Adam(itertools.chain(self.netW_A.parameters(), self.netW_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_W)

        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        AtoB = self.opt.direction == 'AtoB'  # use <direction> to swap data_A and data_B
        self.real_A = input['A' if AtoB else 'B'].to(self.device)  # get image data A
        self.real_B = input['B' if AtoB else 'A'].to(self.device)  # get image data B
        self.image_paths = input['A_paths' if AtoB else 'B_paths']  # get image paths
        self.batch_size = self.real_A.shape[0]

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
            
        self.fake_B = self.netG_A(self.real_A)  # G_xy(x) in the paper
        self.fake_A = self.netG_B(self.real_B)  # G_yx(y) in the paper

        if self.isTrain:
            self.discriminated_A = self.netD(self.real_A, self.fake_B) # D(x, G_xy(x))
            self.discriminated_B = self.netD(self.fake_A, self.real_B) # D(G_yx(y), y)

            self.w_real_A = self.netW_A(self.real_A) # W_x(x)
            self.w_fake_A = self.netW_A(self.fake_A) # W_x(G_yx(y))
            self.w_real_B = self.netW_B(self.real_B) # W_y(y)
            self.w_fake_B = self.netW_B(self.fake_B) # W_y(G_xy(x))
   
            self.weights_A = 0.5*(torch.sigmoid(self.w_real_A) + torch.sigmoid(-self.w_fake_B))
            self.weights_B = 0.5*(torch.sigmoid(-self.w_fake_A) + torch.sigmoid(self.w_real_B))
        
        self.rec_A = self.netG_B(self.fake_B) # G_B(G_A(A))
        self.rec_B = self.netG_A(self.fake_A) # G_A(G_B(B))
        
        self.idt_A = self.netG_B(self.real_A)
        self.idt_B = self.netG_A(self.real_B)


    def compute_Ls(self):
        """Computes L- and L+ of the paper """
        self.L_minus = self.criterionGAN.L_minus(self.discriminated_A, self.weights_A)
        self.L_plus = self.criterionGAN.L_plus(self.discriminated_B, self.weights_B)
        self.loss_minus = self.L_minus # For visualization
        self.loss_plus = self.L_plus   # has to have 'loss' in the variable name

    def backward_G(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.compute_Ls()
        self.loss_G = self.criterionGAN.loss_G(self.L_minus, self.L_plus)
        self.loss_G.backward()
    
    def backward_W(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.compute_Ls()
        self.loss_W = self.criterionGAN.loss_W(self.L_minus, self.L_plus) 
        self.loss_W.backward()

    def backward_D(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.compute_Ls()
        self.loss_D = self.criterionGAN.loss_D(self.L_minus, self.L_plus)
        self.loss_D.backward()

    def optimize_parameters_G(self):
        self.optimizer_G.zero_grad()   # clear networks existing gradients
        self.forward()                 # call forward to calculate intermediate results
        self.backward_G()              # calculate loss and gradients for network G
        self.optimizer_G.step()        # update gradients for network G

    def optimize_parameters_W(self):
        self.optimizer_W.zero_grad()   # clear networks existing gradients
        self.forward()                 # call forward to calculate intermediate results
        self.backward_W()              # calculate loss and gradients for network  W        
        self.optimizer_W.step()        # update gradients for network W

    def optimize_parameters_D(self):
        self.optimizer_D.zero_grad()   # clear networks existing gradients
        self.forward()                 # call forward to calculate intermediate results
        self.backward_D()              # calculate loss and gradients for network D
        self.optimizer_D.step()        # update gradients for network D


    def optimize_parameters(self):
        """Update network weights for G and W; it will be called in every training iteration."""
        self.optimize_parameters_G()
        self.optimize_parameters_D()
        self.optimize_parameters_W()
