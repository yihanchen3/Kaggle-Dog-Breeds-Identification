import os 
import pandas as pd 
import torch 
import torchvision 
from torch import nn 
from d2l import torch as d2l 
import matplotlib.pyplot as plt 
import numpy as np


# use pre_trained model to extract features from dataset, return a list of features and labels
def extract_features(model_name, data_iter, device):
    print('extracting features from', model_name, '...')
    model = getattr(torchvision.models, model_name)(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    feature_extractor = nn.Sequential(*list(model.children())[:-1],
                                      nn.AdaptiveAvgPool2d(output_size=(1, 1)))
    feature_extractor.to(device)
    feature_extractor.eval()
    features = []
    labels = []
    for X, y in data_iter:
        X = X.to(device)
        with torch.no_grad():
            feature = feature_extractor(X).cpu()
        features.append(feature)
        labels.append(y)
    print('model: ',model_name,'features: ', torch.cat(features).shape)
    features = torch.cat(features)
    labels = torch.cat(labels)
    # Flatten the output of the global average pooling layer
    features = features.view(features.shape[0], -1)
    return features, labels


# create a dnn model take features as input
class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x
    
# create a function to calculate the loss of the model
def evaluate_loss(data_iter, net, devices):
    l_sum, n = 0.0, 0
    loss = nn.CrossEntropyLoss()
    for features, labels in data_iter:
        features, labels = features.to(devices[0]), labels.to(devices[0])
        outputs = net(features)
        l = loss(outputs, labels)
        l_sum += l.sum()
        n += labels.numel()
    return l_sum / n

# create a function to train the model
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):

    # set the model to first device
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    # set the loss function and optimizer
    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.Adam(
        (param for param in net.parameters() if param.requires_grad), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss']
    if valid_iter is not None:
        legend.append('valid loss')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(2)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            features, labels = features.to(devices[0]), labels.to(devices[0])
            trainer.zero_grad()
            output = net(features)
            l = loss(output, labels).sum()
            l.backward()
            trainer.step()
            metric.add(l, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1], None))
        measures = f'train loss {metric[0] / metric[1]:.3f}'
        if valid_iter is not None:
            valid_loss = evaluate_loss(valid_iter, net, devices)
            animator.add(epoch + 1, (None, valid_loss.cpu().detach()))
        scheduler.step()
    if valid_iter is not None:
        measures += f', valid loss {valid_loss:.3f}'
    print(measures + f'\n{metric[1] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
    
    fig = plt.gcf()
    os.makedirs('./outputs/figures', exist_ok=True)
    fig.savefig(f'./outputs/figures/epoch_{epoch+1}.png')


# create a function to save the model
def save_model(net, model_name):
    os.makedirs('./outputs/models', exist_ok=True)
    torch.save(net.state_dict(), './outputs/models/' + model_name + '.pt')
    print(model_name + ' model saved')