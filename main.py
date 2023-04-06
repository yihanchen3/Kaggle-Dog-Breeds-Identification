import os 
import pandas as pd 
import torch 
import torchvision 
from torch import nn 
from d2l import torch as d2l 
import matplotlib.pyplot as plt 
import numpy as np
from src.preprocessing import *
from src.visualization import *
from src.model import *
import warnings

warnings.filterwarnings("ignore")

# define the batch size and data loaders
batch_size =  128
valid_ratio = 0.1
# define the transforms for training and testing
transform_train = transform_train_define()
transform_test = transform_test_define()

# reorganize the data into train, valid and test folders
if not os.path.exists(os.path.join('data', 'train_valid_test')):
    reorg_dog_data('data', valid_ratio)

# load the data with the defined transforms
train_ds, train_valid_ds = [
    torchvision.datasets.ImageFolder(
        os.path.join('data', 'train_valid_test', folder),
        transform=transform_train) for folder in ['train', 'train_valid']]

valid_ds, test_ds = [
    torchvision.datasets.ImageFolder(
        os.path.join('data', 'train_valid_test', folder),
        transform=transform_test) for folder in ['valid', 'test']]

print('***data loading finished***')

# create the data loaders for training, validation and testing
train_iter, train_valid_iter = [torch.utils.data.DataLoader(
    dataset, batch_size, shuffle=True, drop_last=True)
    for dataset in (train_ds, train_valid_ds)]

valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False,
                                         drop_last=True)

test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False,
                                        drop_last=False)


# define the best 2 base model names
model_names = ['resnet152','densenet161']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
load = False

if load:
    # load the features from the saved files
    resnet152_features_train = torch.load('outputs/features/resnet152_features_train.pt')
    densenet161_features_train = torch.load('outputs/features/densenet161_features_train.pt')
    resnet152_labels_train = torch.load('outputs/features/resnet152_labels_train.pt')
    densenet161_labels_train = torch.load('outputs/features/densenet161_labels_train.pt')
    resnet152_features_valid = torch.load('outputs/features/resnet152_features_valid.pt')
    densenet161_features_valid = torch.load('outputs/features/densenet161_features_valid.pt')
    resnet152_labels_valid = torch.load('outputs/features/resnet152_labels_valid.pt')
    densenet161_labels_valid = torch.load('outputs/features/densenet161_labels_valid.pt')
    resnet152_features_test = torch.load('outputs/features/resnet152_features_test.pt')
    densenet161_features_test = torch.load('outputs/features/densenet161_features_test.pt')
    resnet152_labels_test = torch.load('outputs/features/resnet152_labels_test.pt')
    densenet161_labels_test = torch.load('outputs/features/densenet161_labels_test.pt')

else:
    # extract features using the base models from dataset
    resnet101_features_train, resnet101_labels_train = extract_features(model_names[0], train_iter, device)
    densenet161_features_train, densenet161_labels_train = extract_features(model_names[1], train_iter, device)

    resnet101_features_valid, resnet101_labels_valid = extract_features(model_names[0], valid_iter, device)
    densenet161_features_valid, densenet161_labels_valid = extract_features(model_names[1], valid_iter, device)

    resnet101_features_test, resnet101_labels_test = extract_features(model_names[0], test_iter, device)
    densenet161_features_test, densenet161_labels_test = extract_features(model_names[1], test_iter, device)

    print('***features extracting finished***')
    print('resnet101_features_train.shape:', resnet101_features_train.shape)
    print('densenet161_features_train.shape:', densenet161_features_train.shape)

    # create features folder and save features
    os.makedirs('./outputs/features', exist_ok=True)
    torch.save(resnet101_features_train, './outputs/features/resnet101_features_train.pt')
    torch.save(resnet101_labels_train, './outputs/features/resnet101_labels_train.pt')
    torch.save(densenet161_labels_train, './outputs/features/densenet161_features_train.pt')
    torch.save(densenet161_features_train, './outputs/features/densenet161_features_train.pt')

    torch.save(resnet101_features_valid, './outputs/features/resnet101_features_valid.pt')
    torch.save(resnet101_labels_valid, './outputs/features/resnet101_labels_valid.pt')
    torch.save(densenet161_features_valid, './outputs/features/densenet161_features_valid.pt')
    torch.save(densenet161_labels_valid, './outputs/features/densenet161_labels_valid.pt')

    torch.save(resnet101_features_test, './outputs/features/resnet101_features_test.pt')
    torch.save(resnet101_labels_test, './outputs/features/resnet101_labels_test.pt')
    torch.save(densenet161_features_test, './outputs/features/densenet161_features_test.pt')
    torch.save(densenet161_labels_test, './outputs/features/densenet161_labels_test.pt')

print('***features saving finished***')

# concatenate features from different models
features_train = torch.cat((resnet101_features_train, densenet161_features_train), dim=1)
print('Final train features shape:', features_train.shape)
features_valid = torch.cat((resnet101_features_valid, densenet161_features_valid), dim=1)
print('Final valid features shape:', features_valid.shape)
features_test = torch.cat((resnet101_features_test, densenet161_features_test), dim=1)
print('Final test features shape:', features_test.shape)

# get the labels corresponding to the features
labels_train = resnet101_labels_train
print('labels_train.shape:', labels_train.shape)

labels_valid = resnet101_labels_valid
print('labels_valid.shape:', labels_valid.shape)

labels_test = resnet101_labels_test
print('labels_test.shape:', labels_test.shape)

# create a dataloader for the features
train_features_ds = torch.utils.data.TensorDataset(features_train, labels_train)
train_features_iter = torch.utils.data.DataLoader(train_features_ds, batch_size=128, shuffle=True)

valid_features_ds = torch.utils.data.TensorDataset(features_valid, labels_valid)
valid_features_iter = torch.utils.data.DataLoader(valid_features_ds, batch_size=128, shuffle=True)  

test_features_ds = torch.utils.data.TensorDataset(features_test, labels_test)
test_features_iter = torch.utils.data.DataLoader(test_features_ds, batch_size=128, shuffle=True)

print('***features dataloader created***')

    
# plot the number of images per breed
plot_images_per_breed('labels.csv')

# plot the raw imagesin the training dataset
show_raw_images(train_iter)

# plot the transformed images
show_transformed_images(transform_train,transform_test)

# train the model
# define the model and set the hyperparameters
net = DNN(input_size=features_train.shape[1:].numel(), hidden_size=1024, output_size=120)
print(net)
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 50, 1e-4, 1e-4
lr_period, lr_decay = 2, 0.75
print('training on', devices)
train(net, train_features_iter, valid_features_iter, num_epochs, lr, wd, devices, lr_period, lr_decay)
save_model(net, 'dnn_train')

# predict the test dataset
preds = []
for data, label in test_features_iter:
    output = torch.nn.functional.softmax(net(data.to(devices[0])), dim=0)
    preds.extend(output.cpu().detach().numpy())
ids = sorted(
    os.listdir(os.path.join('data', 'train_valid_test', 'test', 'unknown')))
with open('./outputs/submission.csv', 'w') as f:
    f.write('id,' + ','.join(train_valid_ds.classes) + '\n')
    for i, output in zip(ids, preds):
        f.write(
            i.split('.')[0] + ',' + ','.join([str(num)
                                            for num in output]) + '\n')

print('***prediction finished***')
