"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import sys

import torch
import ntpath
from data.image_folder import make_dataset
from torchvision.utils import make_grid, save_image
from util import util
from PIL import Image
import numpy as np
from data.base_dataset import BaseDataset, get_transform
import torchvision.transforms as transforms


if __name__ == '__main__':
    #opt = TestOptions().parse()  # get test options
    try:
        name = sys.argv[1]
    except IndexError:
        print("python generate_grid_img.py input_path")
        raise

    dir_A_path = os.path.join("./results", name + "/test_latest/images")  # create a path '/path/to/data/results': dir_A_path = "./results" + opt.name + "/test_lates/images"
    dir_res_save = os.path.join("./results", name + "/test_latest/grid_images") #res_path = "./results" + opt.name + "/test_lates/grid_images"
    print("Reading images from: "+ dir_A_path)
    print("Writing images from: "+ dir_res_save)

    A_paths_all = sorted(make_dataset(dir_A_path))   # load images from '/path/to/data/results'
    A_paths = []
    for p in A_paths_all:
        if "checkpoint" not in p:
            A_paths.append(p)
    transformer = transforms.ToTensor()

    #order images to be displayed:
    test_imgs_paths = []
    amount = 16
    for i in range(0, int(amount)):
        test_imgs_paths_unordered = A_paths[i*8 : i*8+8]
        # #order images to be displayed:
        # test_imgs_paths = []
        #first line: results from real A
        test_imgs_paths.append(test_imgs_paths_unordered[4]) #real A
        # test_imgs_paths.append(test_imgs_paths_unordered[1]) #fake B
        # test_imgs_paths.append(test_imgs_paths_unordered[6]) #rec A
        # test_imgs_paths.append(test_imgs_paths_unordered[2]) #idt A
        # second line: results from real B
        # test_imgs_paths.append(test_imgs_paths_unordered[5]) #real B
        # test_imgs_paths.append(test_imgs_paths_unordered[0]) #fake A
        # test_imgs_paths.append(test_imgs_paths_unordered[7]) #rec B
        # test_imgs_paths.append(test_imgs_paths_unordered[3]) #idt B
    for i in range(0, int(amount)):
        test_imgs_paths_unordered = A_paths[i*8 : i*8+8]
        test_imgs_paths.append(test_imgs_paths_unordered[1]) #fake B
    for i in range(0, int(amount)):
        test_imgs_paths_unordered = A_paths[i*8 : i*8+8]
        test_imgs_paths.append(test_imgs_paths_unordered[6]) #rec A
    for i in range(0, int(amount)):
        test_imgs_paths_unordered = A_paths[i*8 : i*8+8]
        test_imgs_paths.append(test_imgs_paths_unordered[2]) #idt A

    test_img = Image.open(test_imgs_paths[0]).convert('RGB')
    test_img_tensor = transformer(test_img).repeat(1, 1, 1, 1)

    for j in range(1, 16*4):
        img = Image.open(test_imgs_paths[j]).convert('RGB')
        img_tensor = transformer(img).repeat(1, 1, 1, 1)
        test_img_tensor = torch.cat((test_img_tensor, img_tensor), dim = 0)

    if not os.path.exists(dir_res_save):
        os.makedirs(dir_res_save)
    short_path = ntpath.basename(test_imgs_paths_unordered[0])
    short_path = short_path.replace("_fake_A", "")
    name = os.path.splitext(short_path)[0]
    image_name = '%sB_grid.png' % (name)
    image_save_path = os.path.join(dir_res_save, image_name)
    save_image(test_img_tensor, image_save_path, nrow=16, padding=0)

    test_imgs_paths = []
    amount = 16
    for i in range(0, int(amount)):
        test_imgs_paths_unordered = A_paths[i*8 : i*8+8]
        test_imgs_paths.append(test_imgs_paths_unordered[5]) #real A
    for i in range(0, int(amount)):
        test_imgs_paths_unordered = A_paths[i*8 : i*8+8]
        test_imgs_paths.append(test_imgs_paths_unordered[0]) #fake B
    for i in range(0, int(amount)):
        test_imgs_paths_unordered = A_paths[i*8 : i*8+8]
        test_imgs_paths.append(test_imgs_paths_unordered[7]) #rec A
    for i in range(0, int(amount)):
        test_imgs_paths_unordered = A_paths[i*8 : i*8+8]
        test_imgs_paths.append(test_imgs_paths_unordered[3]) #idt A

    test_img = Image.open(test_imgs_paths[0]).convert('RGB')
    test_img_tensor = transformer(test_img).repeat(1, 1, 1, 1)

    for j in range(1, 16*4):
        img = Image.open(test_imgs_paths[j]).convert('RGB')
        img_tensor = transformer(img).repeat(1, 1, 1, 1)
        test_img_tensor = torch.cat((test_img_tensor, img_tensor), dim = 0)

    if not os.path.exists(dir_res_save):
        os.makedirs(dir_res_save)
    short_path = ntpath.basename(test_imgs_paths_unordered[0])
    short_path = short_path.replace("_fake_A", "")
    name = os.path.splitext(short_path)[0]
    image_name = '%sA_grid.png' % (name)
    image_save_path = os.path.join(dir_res_save, image_name)
    save_image(test_img_tensor, image_save_path, nrow=16, padding=0)
