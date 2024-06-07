import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from utils import *
from torch.nn.utils.rnn import pad_sequence
from gradcam import GradCam
from PIL import Image
import requests
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoProcessor, CLIPModel
from torchvision import transforms
from PIL import Image
import torch
from torch import nn, optim
from transformers import CLIPModel
from vit_att import *

import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Process the task type.')

parser.add_argument('--task', type=str, default='object', choices=['action', 'object'],
                    help='Specify the task type: "action" or "object".')
parser.add_argument('--num_epochs', type=int, default=10,
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

transform_image = transforms.Compose([
                  transforms.Resize((224, 224)),
                  transforms.ToTensor(),
                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]) 
        
transform_visualexp = transforms.Compose([
                      transforms.Resize((224, 224)),
                      transforms.ToTensor()
                      ]) 

# Assuming CustomDataset is defined somewhere
dataset = MMESDataset(data, root_dir, transform_image, transform_visualexp)

# Split the dataset into train and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# DataLoaders for train and test datasets
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_fn)

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")


clip_vision_transformer = model.vision_model  # Assuming you have the original CLIPVisionTransformer model defined
# Create an instance of the classifier model
model_cls = CLIPVisionTransformerClassifier(clip_vision_transformer, num_classes)

# Assuming you have already set up the model_cls from the provided class definition
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_cls.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_cls.parameters(), lr=1e-4)


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in dataloader:
        images = data['image'].to(device)
        labels = data['class_id'].to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate losses
        total_loss += loss.item()

    # Calculate average loss
    avg_total_loss = total_loss / len(dataloader)
    
    return avg_total_loss
ges = data['image'].to(device)
    

def test(model, dataloader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient computation
        for data in dataloader:
            images = data['image'].to(device)
            labels = data['class_id'].to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


num_epochs = args.num_epochs
# Training the model
for epoch in range(num_epochs):
    avg_total_loss = train(model_cls, train_loader, optimizer, criterion, device)
    print(f'Epoch [{epoch+1}/{num_epochs}], Total Loss: {avg_total_loss:.4f}')

test_loss, test_accuracy = test(model_cls, test_loader, criterion, device)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')


