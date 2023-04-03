import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import os
import timm

torch.cuda.empty_cache()

# Define the list of pre-trained models
models_list = [  'vgg19',
               'densenet121', 'densenet161', 'densenet169', 'densenet201', 
               'squeezenet1_0', 'squeezenet1_1', 
               'inception_v3','xception','nasnetalarge','inceptionresnetv2' ]

# models_list = ['squeezenet1_1']'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152','vgg11', 'vgg13', 'vgg16',

# Define the custom dataset
data_dir = os.path.join('data', 'dog-breed-identification/train_valid_test')
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'valid')
test_dir = os.path.join(data_dir, 'test')

# Define the data transforms
# Define the transforms for the dataset
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


transform_train_299 = transforms.Compose([
    transforms.RandomResizedCrop(299),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_valid_299 = transforms.Compose([
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the custom dataset
train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
val_dataset = datasets.ImageFolder(val_dir, transform=transform_val)
# test_dataset = datasets.ImageFolder(test_dir, transform=transform)

# Define the batch size for the dataloaders
batch_size = 32

# Create the dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


train_dataset_299 = datasets.ImageFolder(train_dir, transform=transform_train_299)
val_dataset_299 = datasets.ImageFolder(val_dir, transform=transform_valid_299)
train_dataloader_299 = DataLoader(train_dataset_299, batch_size=batch_size, shuffle=True)
val_dataloader_299 = DataLoader(val_dataset_299, batch_size=batch_size, shuffle=False)



# Define the device to use (GPU if available, otherwise CPU)
device = ('cuda' if torch.cuda.is_available() else 'cpu')

# Define the list to store the validation losses
losses = []

# Loop through the list of models
for model_name in models_list:
    print(f'Testing model {model_name}')
    # Load the pre-trained model

    if model_name == 'xception':
        model = getattr(timm.models, model_name)(pretrained=True)
    
    elif model_name == 'inception_resnet_v2':
        model = getattr(timm.models, model_name)(pretrained=True)
    
    elif model_name == 'nasnetalarge':
        model = getattr(timm.models, model_name)(pretrained=True)
    else:
        model = getattr(torchvision.models, model_name)(pretrained=True)
    # Freeze all the layers
    for param in model.parameters():
        param.requires_grad = False
    # Replace the last layer with a new one that fits the custom dataset
    num_classes = len(train_dataset.classes)

    # special case for vgg16
    if 'vgg' in model_name:
        in_features = model.classifier[-1].in_features
        # model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    
    elif 'densenet' in model_name:
        in_features = model.classifier.in_features
        # model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.classifier = nn.Linear(in_features, num_classes)
    # special case for squeezenet1_0
    elif 'squeezenet' in model_name :
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model.num_classes = num_classes
    
    # special case for inceptionresnetv2
    elif model_name == 'nasnetalarge' :
        in_features = model.last_linear.in_features
        # model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.last_linear = nn.Linear(in_features, num_classes)  
    
    # special case for inceptionresnetv2
    elif model_name == 'inception_resnet_v2' :
        print(model)
        in_features = model.classif.in_features
        # model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.classif = nn.Linear(in_features, num_classes)

    else:
        in_features = model.fc.in_features
        # Add a global average pooling layer and a fully connected layer to the model
        # model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.fc = nn.Linear(in_features, num_classes)

    # print(model)

    # Move the model to the device
    model = model.to(device)
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters())
    # Train the model for one epoch on the custom dataset

    num_epochs = 50
    for epoch in range(num_epochs):
        # Train the model
        model.train()
        train_loss = 0.0
        train_correct = 0

        if model_name == 'inception_v3':
            train_dataloader = train_dataloader_299
            val_dataloader = val_dataloader_299


        for inputs, labels in tqdm(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            if model_name == 'inception_v3':
                outputs,aux = model(inputs)
            else:
                outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)
        train_loss = train_loss / len(train_dataset)
        train_acc = train_correct.double() / len(train_dataset)
        # Evaluate the model on the validation set
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)
        val_loss = val_loss / len(val_dataset)
        val_acc = val_correct.double() / len(val_dataset)

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Model: {model_name}')
    # store val_acc and model_name in a list
    losses.append((model_name, train_loss,train_acc, val_loss, val_acc))

# rank the models by val_loss
df = pd.DataFrame(losses, columns=['model', 'train_loss','train_acc', 'val_loss', 'val_acc'])
df = df.sort_values('val_loss')
save_path = './'
df.to_csv(save_path + 'models_compare.csv')
print(df)


    # # Evaluate the model on the test set
    # model.eval()
    # test_loss = 0.0
    # test_correct = 0
    # with torch.no_grad():
    #     for inputs, labels in tqdm(test_dataloader):
    #         inputs = inputs.to(device)
    #         labels = labels.to(device)
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #         test_loss += loss.item() * inputs.size(0)
    #         _, preds = torch.max(outputs, 1)
    #         test_correct += torch.sum(preds == labels.data)
    # test_loss = test_loss / len(test_dataset)
    # test_acc = test_correct.double() / len(test_dataset)
    # print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
