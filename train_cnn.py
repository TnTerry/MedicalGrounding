import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import json
import torch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from utils import *
from torch.nn.utils.rnn import pad_sequence

import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Process the task type.')

parser.add_argument('--task', type=str, required=True, choices=['action', 'object'],
                    help='Specify the task type: "action" or "object".')
parser.add_argument('--num_epochs', type=int, default=30,
                    help='Specify the number of epochs for training (default: 30).')

# Parse the command-line arguments
args = parser.parse_args()

# Get the directory path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Set the current working directory to the script directory
os.chdir(script_dir)
print("Current Working Directory: ", os.getcwd())

# Assuming data is defined properly
def load_json_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

if args.task == "action":
    data = load_json_data("Datasets/Action_Classification/exp_annotation.json")
    root_dir = 'Datasets/Action_Classification/'  # Adjust as necessary to point to the correct root directory

    id_map = load_json_data("Datasets/Action_Classification/label_class_id_mapping.json")
elif args.task == "object":
    data = load_json_data("Datasets/Object_Classification/exp_annotation.json")
    root_dir = 'Datasets/Object_Classification/'  # Adjust as necessary to point to the correct root directory

    id_map = load_json_data("Datasets/Object_Classification/label_class_id_mapping.json")
else:
    raise ValueError("Invalid task specified. Expected 'action' or 'object'.")


# Load a pre-trained ResNet-50 model and modify it for your number of classes
num_classes = len(id_map)  # Update this to your actual number of classes

def custom_collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    visual_exps = torch.stack([item['visual_exp'] for item in batch])
    class_ids = torch.tensor([item['class_id'] for item in batch])
    exps = [item['exp'] for item in batch]  # List of lists

    # If you're using tokenized indices for text and want to pad:
    # exps_padded = pad_sequence([torch.tensor(exp) for sublist in exps for exp in sublist], 
    #                            batch_first=True, padding_value=0)

    return {'image': images, 'visual_exp': visual_exps, 'class_id': class_ids, 'exp': exps}

# Transformation pipeline including conversion to tensor
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transformation for the images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Assuming CustomDataset is defined somewhere
dataset = MMESDataset(data, root_dir, transform=transform)

# Split the dataset into train and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# DataLoaders for train and test datasets
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_fn)
    
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Check if GPU is available and move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images = data['image'].to(device)
        labels = data['class_id'].to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the model on the test images before fine-tuning: {accuracy:.2f}%')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training Loop
for epoch in range(args.num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    
    for i, data in enumerate(train_loader):
        inputs = data['image'].to(device)
        labels = data['class_id'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

    print('Finished Training')

# Testing Loop
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images = data['image'].to(device)
        labels = data['class_id'].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the model on the test images after fine-tuning: {accuracy:.2f}%')