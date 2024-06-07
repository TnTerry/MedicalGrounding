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
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import prettytable as pt
import cv2

import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Process the task type.')

parser.add_argument('--task', type=str, default='action', choices=['action', 'object'],
                    help='Specify the task type: "action" or "object".')
parser.add_argument('--num_epochs', type=int, default=5,
                    help='Specify the number of epochs for training (default: 10).')

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
class_id_lst = [int(dataset[i]['class_id']) for i in range(len(dataset))]
class_id_cnt = Counter(class_id_lst)
print(class_id_cnt)
num_classes = len(class_id_cnt)

# if args.task == "action":    
#     class_id_transform_dict = {}
#     flag = 0
#     for class_id in class_id_cnt.keys():
#         class_id_transform_dict[class_id] = flag
#         flag += 1

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
BCE_criterion = nn.BCELoss()
l1_criterion = nn.L1Loss(reduction="none")
optimizer = optim.Adam(model_cls.parameters(), lr=1e-4)


def BF_solver(X, Y):
    epsilon = 1e-4

    with torch.no_grad():
        x = torch.flatten(X)
        y = torch.flatten(Y)
        g_idx = (y<0).nonzero(as_tuple=True)[0]
        le_idx = (y>0).nonzero(as_tuple=True)[0]
        len_g = len(g_idx)
        len_le = len(le_idx)
        a = 0
        a_ct = 0.0
        for idx in g_idx:
            v = x[idx] + epsilon # to avoid miss the constraint itself
            v_ct = 0.0
            for c_idx in g_idx:
                v_ct += (v>x[c_idx]).float()/len_g
            for c_idx in le_idx:
                v_ct += (v<=x[c_idx]).float()/len_le
            if v_ct>a_ct:
                a = v
                a_ct = v_ct
                # print('New best solution found, a=', a, ', # of constraints matches:', a_ct)

        for idx in le_idx:
            v = x[idx]
            v_ct = 0.0
            for c_idx in g_idx:
                v_ct += (v>x[c_idx]).float()/len_g
            for c_idx in le_idx:
                v_ct += (v<=x[c_idx]).float()/len_le
            if v_ct>a_ct:
                a = v
                a_ct = v_ct
                # print('New best solution found, a=', a, ', # of constraints matches:', a_ct)

    # print('optimal solution for batch, a=', a)
    # print('final threshold a is assigned as:', am)

    return torch.tensor([a]).cuda()

def visual_exp_transform(visual_exp_map, trans_type=None):
    if not trans_type or trans_type == "HAICS" or trans_type == "GRADIA":
        return visual_exp_map
    
    if trans_type == "Gaussian":
        visual_exp_map_pos = np.maximum(visual_exp_map.cpu().numpy(), 0)
        visual_exp_map_trans = cv2.GaussianBlur(visual_exp_map_pos, (3,3), 0) # Gaussian Blur
        visual_exp_map_trans = visual_exp_map_trans / (np.max(visual_exp_map_trans)+1e-6)
    
    return torch.from_numpy(visual_exp_map_trans).cuda()
    

def cal_trans_att_loss(att_map, visual_exp_trans, trans_type=None):
    if not trans_type:
        trans_att_loss = 0
    
    if trans_type == "Gaussian":
        a = BF_solver(att_map, visual_exp_trans)
        temp1 = torch.tanh(5*(att_map - a))
        temp_loss = F.l1_loss(temp1, visual_exp_trans, reduction="mean")
        temp_size = torch.count_nonzero(visual_exp_trans, dim=(1, 2)).float()
        eff_loss = torch.sum(temp_loss * temp_size) / torch.sum(temp_size)
        trans_att_loss = torch.relu(torch.mean(eff_loss) - 0)

        # att_map_labels_trans = torch.stack(visual_exp_trans)
        tempD = F.l1_loss(att_map, visual_exp_trans)
        trans_att_loss = trans_att_loss + tempD
    
    elif trans_type == "HAICS":
        temp_att_loss = BCE_criterion(att_map, visual_exp_trans * (visual_exp_trans > 0).float())
        mask = torch.count_nonzero(visual_exp_trans, dim=0).float()
        trans_att_loss = torch.mean(temp_att_loss * mask)
    elif trans_type == "GRADIA":
        mask = torch.count_nonzero(visual_exp_trans, dim=0).float()
        temp_att_loss = l1_criterion(att_map, visual_exp_trans * mask)
        trans_att_loss = torch.mean(temp_att_loss)
    
    return trans_att_loss


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_pred_loss = 0
    total_att_loss = 0
    for i, data in enumerate(dataloader):
        images = data['image'].to(device)
        labels = data['class_id'].to(device)
        # if args.task == "action":
        #     labels_lst_form = labels.cpu().numpy().tolist()
        #     labels_new = [class_id_transform_dict[id_label] 
        #                   for id_label in labels_lst_form]
        #     labels = torch.tensor(labels_new).to(device)
        visual_exps = data['visual_exp'].to(device)

        # Forward pass
        outputs = model(images)
        pred_loss = criterion(outputs, labels)
        
        attn_weights = model.get_attention_map() # torch.Size([16, 50, 50])

        visual_exps_resized = resize_vit_att(visual_exps, attn_weights.shape[1], attn_weights.shape[2]).to(device)
        att_loss = F.l1_loss(visual_exps_resized, attn_weights, reduction='mean') * 100 # Original att loss

        # Transformation att loss
        visual_exp_resized_trans = visual_exp_transform(visual_exps_resized, "GRADIA")
        att_loss = att_loss + cal_trans_att_loss(attn_weights, visual_exp_resized_trans, "GRADIA")
        
        loss = pred_loss + att_loss
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate losses
        total_loss += loss.item()
        total_pred_loss += pred_loss.item()
        total_att_loss += att_loss.item()

        if i % 100 == 0:
            print(f"Round {i}: Total loss: {loss.item()}, Pred loss: {pred_loss.item()}, Att loss: {att_loss.item()}")

    # Calculate averages
    avg_total_loss = total_loss / len(dataloader)
    avg_pred_loss = total_pred_loss / len(dataloader)
    avg_att_loss = total_att_loss / len(dataloader)
    
    return avg_total_loss, avg_pred_loss, avg_att_loss

def test(model, dataloader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    correct = 0
    total = 0
    total_label = []
    total_pred = []
    total_outputs = []
    
    with torch.no_grad():  # Disable gradient computation
        for data in dataloader:
            images = data['image'].to(device)
            labels = data['class_id'].to(device)
            # if args.task == "action":
            #     labels_lst_form = labels.cpu().numpy().tolist()
            #     labels_new = [class_id_transform_dict[id_label] 
            #                 for id_label in labels_lst_form]
            #     labels = torch.tensor(labels_new).to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_label.extend(labels.cpu().tolist())
            total_pred.extend(predicted.cpu().tolist())
            total_outputs.extend(outputs.cpu().tolist())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total

    try:
        outputs_cnt, label_cnt = Counter(total_outputs), Counter(total_label)
        print(outputs_cnt, label_cnt)
    except:
        pass

    total_outputs = np.array(total_outputs)
    total_label = np.array(total_label)

    # Evaluation Metrics Computation
    test_accuracy_raw = accuracy_score(total_label, total_pred)
    test_recall_raw_micro = recall_score(total_label, total_pred, average="micro")
    test_precision_raw_micro = precision_score(total_label, total_pred, average="micro")
    test_f1_raw_micro = f1_score(total_label, total_pred, average="micro")

    soft_max_outputs = torch.tensor(total_outputs)
    soft_max_outputs = F.softmax(soft_max_outputs)

    test_auc_raw_micro = roc_auc_score(total_label, soft_max_outputs.numpy(), average="micro", multi_class="ovr")

    test_recall_raw_macro = recall_score(total_label, total_pred, average="macro")
    test_precision_raw_macro = precision_score(total_label, total_pred, average="macro")
    test_f1_raw_macro = f1_score(total_label, total_pred, average="macro")
    test_auc_raw_macro = roc_auc_score(total_label, soft_max_outputs.numpy(), average="macro", multi_class="ovr")

    print(confusion_matrix(total_label, total_pred))

    tb = pt.PrettyTable()
    tb.field_names = ["", "Accuracy", "Recall", "Precision", "F1", "AUC"]
    tb.add_row(
        ["Micro",test_accuracy_raw, test_recall_raw_micro, 
        test_precision_raw_micro, test_f1_raw_micro, test_auc_raw_micro]
    )
    tb.add_row(
        ['Macro', test_accuracy_raw, test_recall_raw_macro, 
        test_precision_raw_macro, test_f1_raw_macro, test_auc_raw_macro]
    )
    print(tb)

    return avg_loss, accuracy


num_epochs = args.num_epochs
# Training the model
for epoch in range(num_epochs):
    avg_total_loss, avg_pred_loss, avg_l1_loss = train(model_cls, train_loader, optimizer, criterion, device)
    print(f'Epoch [{epoch+1}/{num_epochs}], Total Loss: {avg_total_loss:.4f}, Prediction Loss: {avg_pred_loss:.4f}, Attention Loss: {avg_l1_loss:.4f}')
    
test_loss, test_accuracy = test(model_cls, test_loader, criterion, device)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')


