import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split

def get_model_features(model, input_size, data):
    '''
    1- Create a feature extractor to extract features from the data.
    2- Returns the extracted features and the feature extractor.
    '''
    # Prepare pipeline.
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    # Extract features.
    features = []
    for x in data:
        with torch.no_grad():
            x = x.unsqueeze(0)
            x = x.to(device)
            feature = feature_extractor(x).squeeze()
            feature = feature.cpu().numpy()
            features.append(feature)
    features = np.array(features)
    print('Features shape:', features.shape)
    return features

# Load and preprocess data
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data = ImageFolder(root='./data', transform=data_transforms)

# Split data into train and validation sets
train_data, val_data, train_labels, val_labels = train_test_split(data.data, data.targets, test_size=0.2)

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define models to evaluate
model_names = ['resnet18', 'resnet50', 'densenet121', 'densenet169', 'vgg16']

# Evaluate models and save results
results = []
for model_name in model_names:
    print(f'Evaluating {model_name}...')
    # Load pre-trained model
    model = models.__dict__[model_name](pretrained=True)
    model = model.to(device)
    # Extract features from validation set
    val_features = get_model_features(model, (3, 224, 224), val_data)
    # Define DNN model
    dnn = nn.Sequential(
        nn.Linear(val_features.shape[1], 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, data.classes)
    )
    dnn = dnn.to(device)
    # Compile model
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(dnn.parameters(), lr=0.001)
    # Train model
    num_epochs = 10
    batch_size = 32
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0
        for i in range(0, len(train_data), batch_size):
            # Get batch of data and labels
            batch_data = train_data[i:i+batch_size]
            batch_labels = train_labels[i:i+batch_size]
            # Extract features from batch
            batch_features = get_model_features(model, (3, 224, 224), batch_data)
            # Forward pass
            outputs = dnn(torch.tensor(batch_features).to(device))
            loss = criterion(outputs, torch.tensor(batch_labels).to(device))
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Update statistics
            running_loss += loss.item() * batch_features.shape[0]
            running_corrects += torch.sum(torch.argmax(outputs, 1) == torch.tensor(batch_labels).to(device))
        epoch_loss = running_loss / len(train_data)
        epoch_acc = running_corrects.double() / len(train_data)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
    # Evaluate model
    with torch.no_grad():
        val_features = get_model_features(model, (3, 224, 224), val_data)
        outputs = dnn(torch.tensor(val_features).to(device))
        _, preds = torch.max(outputs, 1)
        acc = torch.sum(preds == torch.tensor(val_labels).to(device)).double() / len(val_labels)
        print(f'Accuracy: {acc:.4f}')
    results.append(acc)

# Print results
for model_name, acc in zip(model_names, results):
    print(f'{model_name}: {acc:.4f}')

# sort the results and get the best 4 models
best_models = [model_names[i] for i in np.argsort(results)[::-1][:4]]

# print the best models wit their accuracy
print('Best models:')
for model_name, acc in zip(model_names, results):
    if model_name in best_models:
        print(f'{model_name}: {acc:.4f}')

# save the best models
for model_name in best_models:
    model = models.__dict__[model_name](pretrained=True)
    model = model.to(device)
    torch.save(model.state_dict(), f'./models/{model_name}.pth')

    


