import os 
import pandas as pd 
import torch 
import torchvision 
from torch import nn 
from d2l import torch as d2l 
import matplotlib.pyplot as plt 
import numpy as np
import tabulate
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image

# denormalize images after data augmentation
def imshow(axis, inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    axis.imshow(inp)

# display the raw images after data augmentation
def show_raw_images(train_iter):
    img, label = next(iter(train_iter))
    fig = plt.figure(1, figsize=(16, 4))
    grid = ImageGrid(fig, 111, nrows_ncols=(3, 4), axes_pad=0.05)    
    for i in range(12):
        ax = grid[i]
        img_data = img[i]
        imshow(ax, img_data)
    fig.savefig(os.path.join('outputs','figures','raw_images.png'))

# compare the image before and after transform
def show_transformed_images(transform_train, transform_test):
    img_train = Image.open('./data/train/0a0c223352985ec154fd604d7ddceabd.jpg')
    img_test = Image.open('./data/test/0a0b97441050bba8e733506de4655ea1.jpg')
    # Apply transform to image
    transformed_img_train = transform_train(img_train)
    transformed_img_test = transform_test(img_test)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(img_train)
    axs[0].set_title('Original Image')
    axs[1].imshow(transformed_img_train.permute(1, 2, 0))
    axs[1].set_title('Transformed Image')
    plt.savefig(os.path.join('outputs','figures', 'transformed_train_image.png'))


    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(img_test)
    axs[0].set_title('Original Image')
    axs[1].imshow(transformed_img_test.permute(1, 2, 0))
    axs[1].set_title('Transformed Image')
    plt.savefig(os.path.join('outputs','figures', 'transformed_test_image.png'))

