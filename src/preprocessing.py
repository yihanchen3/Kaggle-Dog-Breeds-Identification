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



# read the labels.csv file
def read_csv_label(fname):
    with open(fname, 'r') as f:
        # skip the header line
        lines = f.readlines()[1:]
        tokens = [l.rstrip().split(',') for l in lines]
        # seperate the id and label, and return a dictionary with id as key and label as value
        return dict(((id, label) for id, label in tokens))

# plot the number of images per breed
def plot_images_per_breed(labels):
    df = pd.read_csv(os.path.join('data', labels))
    # count the number of images per breed
    counts = df['breed'].value_counts()
    plt.bar(counts.index, counts.values)
    plt.xticks([counts.index[i] for i in range(0,len(counts),5)], rotation = 90 ,fontsize=8)
    plt.title('Number of images per breed')
    plt.xlabel('Breed')
    plt.ylabel('Number of images')
    plt.savefig(os.path.join('outputs','figures','images_per_breed.png'))
    table = pd.DataFrame({'Breed': counts.index, 'Number of images': counts.values})
    print(tabulate.tabulate(table, headers='keys', tablefmt='psql'))


# reorganize the data into train, valid and test folders
def reorg_dog_data(data_dir, valid_ratio):
    labels = d2l.read_csv_labels(os.path.join(data_dir, 'labels.csv'))
    d2l.reorg_train_valid(data_dir, labels, valid_ratio)
    d2l.reorg_test(data_dir)

# define the image augmentation functions
def transform_train_define(resize=224):
    transform_train = torchvision.transforms.Compose([
        # random crop the image to 224x224 size, 
        # scale the image to 0.08 to 1 times of the original size, and the aspect ratio is between 3/4 and 4/3
        torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                                ratio=(3.0 / 4.0, 4.0 / 3.0)),
        # random flip the image horizontally
        torchvision.transforms.RandomHorizontalFlip(),
        # random change the brightness, contrast and saturation
        torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.4),
        # convert the image from PIL image to tensor
        torchvision.transforms.ToTensor(),
        # normalize the image with the mean and standard deviation of the ImageNet dataset
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])])
    return transform_train

def transform_test_define(resize=224):
    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        # crop the image to 224x224 size
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])])
    return transform_test

