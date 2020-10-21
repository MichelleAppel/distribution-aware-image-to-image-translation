import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import network

class WeightModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout

        return parser

    def __init__(self, opt, criterion):
        """Initialize the CycleGAN class.

        Parameters:
            objective_function: a function that can score the performance of the weight network
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        self.criterion = criterion
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['W', 'w0', 'w1', 'wu0', 'wu1', 'count0', 'count1']
        self.model_names = ['W']

        self.netW = networks.WeightNet(opt.input_nc).cuda()
      
        self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
        self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images

        # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
        self.optimizer_W = torch.optim.Adam(self.netW.parameters(), lr=opt.lr)
        self.optimizers.append(self.optimizer_W)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.label_A = input['A_targets' if AtoB else 'B_targets'].to(self.device)
        self.label_B = input['B_targets' if AtoB else 'A_targets'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.w, self.unnorm_w = self.netW(self.real_A)

    def backward_W(self):
        self.loss_W = self.criterion(self)
        self.loss_W.backward() 

    def get_current_examples(self, dataset):
        example_data = dataset.example().cuda()
        example_w, example_w_unnorm = self.netW(example_data)
        self.loss_w0 = example_w[0]
        self.loss_w1 = example_w[1]
        self.loss_wu0 = example_w_unnorm[0]
        self.loss_wu1 = example_w_unnorm[1]
        self.loss_count0 = (self.label_A == 0).sum()/float(self.label_A.shape[0])
        self.loss_count1 = (self.label_A == 1).sum()/float(self.label_A.shape[0])

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
    
        if self.opt.train_W:
            self.optimizer_W.zero_grad()
            self.backward_W()
            self.optimizer_W.step()
