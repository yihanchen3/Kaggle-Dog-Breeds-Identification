import torch
from torch import utils, nn
from torch.utils import data
import torchvision
from torchvision import transforms, datasets, models
import pandas as pd
import os
import random
import time
import shutil
device = ('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = './data/dog-breed-identification'  #下载数据集目录
label_file, train_dir, test_dir = 'labels.csv', 'train', 'test'  # 文件夹data_dir中的文件及子文件夹
new_data_dir = './train_valid_test'  # 整理数据后存放的文件夹
valid_ratio = 0.1  # 验证集所占比例


def mkdir_if_not_exist(path):
    if not os.path.exists(os.path.join(*path)):
        os.makedirs(os.path.join(*path))


def reorg_dog_data(data_dir, label_file, train_dir, test_dir, new_data_dir, valid_ratio):
    # 读取训练数据标签
    labels = pd.read_csv(os.path.join(data_dir, label_file)) # labels是一个列表 labels.size = ([10022,2])
    id2label = {id: value for id, value in labels.values} # id2label是一个字典，其中键值对为id-class

    # 随机打乱训练数据
    train_files = os.listdir(os.path.join(data_dir, train_dir)) # train_files是个列表
    random.shuffle(train_files)

    # 源训练集
    valid_size = int(len(train_files) * valid_ratio)
    for i, file in enumerate(train_files):
        img_id = file.split('.')[0]  # file是形式为id.jpg的字符串
        img_label = id2label[img_id]
        if i < valid_size:
            mkdir_if_not_exist([new_data_dir, 'valid', img_label])
            shutil.copy((os.path.join(data_dir, train_dir, file)), (os.path.join(new_data_dir, 'valid', img_label)))
        else:
            mkdir_if_not_exist([new_data_dir, 'train', img_label])
            shutil.copy((os.path.join(data_dir, train_dir, file)), os.path.join(new_data_dir, 'train', img_label))
        mkdir_if_not_exist([new_data_dir, 'train_valid', img_label])
        shutil.copy(os.path.join(data_dir, train_dir, file), os.path.join(new_data_dir, 'train_valid', img_label))

    # 测试集
        mkdir_if_not_exist([new_data_dir, 'test', 'unknown'])
        for test_file in os.listdir(os.path.join(data_dir, test_dir)):
            shutil.copy(os.path.join(data_dir, test_dir, test_file), os.path.join(new_data_dir, 'test', 'unknown'))

# 进行数据预处理（此过程在普通laptop上运行时间较长，本人运行一下午后放弃了。。。
reorg_dog_data(data_dir, label_file, train_dir, test_dir, new_data_dir, valid_ratio)


# 图像增强

transform_train = transforms.Compose([
                        transforms.RandomResizedCrop(224, scale=(0.1, 1.0)),
                        transforms.RandomHorizontalFlip(0.5),
                        transforms.ColorJitter(0.5, 0.5, 0.5),
                        transforms.ToTensor(),
                    # 对三个通道做标准化，(0.485, 0.456, 0.406)和(0.229, 0.224, 0.225)是在ImageNet上计算得的各通道均值与方差
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet上的均值和方差
                    ])
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# 加载数据集
# new_data_dir目录下有train, valid, train_valid, test四个目录
# 这四个目录中，每个子目录表示一种类别，目录中是属于该类别的所有图像
train_ds = datasets.ImageFolder(root=os.path.join(new_data_dir, 'train'), transform=transform_train)
valid_ds = datasets.ImageFolder(root=os.path.join(new_data_dir, 'valid'), transform=transform_test)
train_valid_ds = datasets.ImageFolder(root=os.path.join(new_data_dir, 'train_valid'), transform=transform_train)
test_ds = datasets.ImageFolder(root=os.path.join(new_data_dir, 'test'), transform=transform_test)
batch_size = 128
train_iter = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, shuffle=True)
train_valid_iter = torch.utils.data.DataLoader(train_valid_ds, batch_size=batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)


# FineTuning GoogleNet模型
def get_net(device):
    # 指定pretrained=True来自动下载并加载预训练的模型参数。在第一次使用时需要联网下载模型参数。
    finetune_net = models.googlenet(pretrained=True)
    finetune_net.fc = nn.Sequential(nn.Linear(in_features=1024, out_features=120, bias=True))
    return finetune_net

# 定义训练函数
def evaluate_loss_acc(data_iter, net, device):
    # 计算data_iter上的平均损失与准确率
    loss = nn.CrossEntropyLoss()
    is_training = net.training
    net.eval()
    l_sum, acc_sum, n = 0.0, 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l_sum += l.item()
            _, predicted = torch.max(y_hat.data, dim=1)
            acc_sum += predicted.eq(y.data).sum().item()
            n += y.shape[0]
    net.train(is_training)  # 恢复net的train/eval状态
    return l_sum / n, acc_sum / n

def train(net, train_iter, valid_iter, num_epochs, lr, wd, device, lr_period, lr_decay):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.fc.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    net = net.to(device)
    for epoch in range(num_epochs):
        train_l_sum, n, start = 0.0, 0, time.time()

        # 设置学习率lr衰减
        if epoch > 0 and epoch % lr_period == 0:  # 每lr_period个epoch，学习率衰减一次
            lr = lr * lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            train_l_sum +=l.item()
            n += y.shape[0]
        time_s = ('time %.2f sec' %(time.time()-start))
        if valid_iter is not None:
            valid_loss, valid_acc = evaluate_loss_acc(valid_iter, net, device)
            epoch_s = ("epoch %d, train loss %f, valid loss %f, valid acc %f, " % (epoch + 1, train_l_sum / n, valid_loss, valid_acc))
        else:
            epoch_s = ('epoch %d, train loss %f' % (epoch + 1, train_l_sum / n))
        print(epoch_s + time_s + ', lr ' + str(lr))


if __name__ == '__main__':
    # 参数设置并调整
    num_epochs, lr_period, lr_decay = 20, 10, 0.1
    lr, wd = 0.03, 1e-4
    net = get_net(device)

    train(net, train_iter, valid_iter, num_epochs, lr, wd, device, lr_period, lr_decay)

    # 通过valid_iter调整好参数后：在完整数据集上训练模型
    train(net, train_valid_iter, None, num_epochs, lr, wd, device, lr_period, lr_decay)

    # 测试并将结果保存为所要求的格式
    preds=[]
    for X, _ in test_iter:
        X = X.to(device)
        output = net(X)
        output = torch.softmax(output, dim=1)
        preds += output.tolist()
    ids = sorted(os.listdir(os.path.join(new_data_dir, 'test/unknown')))

    with open('submission.csv', 'w') as f:
        f.write('id,' + ','.join(train_valid_ds.classes) + '\n')
        for i, output in zip(ids, preds):
            f.write(i.split('.')[0] + ',' + ','.join(
                [str(num) for num in output]) + '\n')
