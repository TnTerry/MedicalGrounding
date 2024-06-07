import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset, SubsetRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import json
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
from utils import *
from torch.nn.utils.rnn import pad_sequence
from gradcam import GradCam
import cv2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import prettytable as pt
from collections import Counter
import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Process the task type.')

parser.add_argument('--task', type=str, default='action', choices=['action', 'object'],
                    help='Specify the task type: "action" or "object".')
parser.add_argument('--num_epochs', type=int, default=4,
                    help='Specify the number of epochs for training (default: 10).')
parser.add_argument('--att_weight', type=float, default=1,
                    help='Weight of the attention loss term in the loss function (default: 1).')

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
                  transforms.Resize((256, 256)),
                  transforms.ToTensor(),
                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]) 
        
transform_visualexp = transforms.Compose([
                      transforms.Resize((256, 256)),
                      transforms.ToTensor()
                      ]) 

# Assuming CustomDataset is defined somewhere
dataset = MMESDataset(data, root_dir, transform_image, transform_visualexp)

# Adjust the num_classes based on the actual dataset
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
# train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_indices, test_indices = train_test_split(
    range(len(dataset)),
    test_size = 0.2
)

train_dataset = Subset(data, train_indices)
test_dataset = Subset(data, test_indices)
sr_sampler_train = SubsetRandomSampler(train_indices)
sr_sampler_test = SubsetRandomSampler(test_indices)


# DataLoaders for train and test datasets
train_loader = DataLoader(dataset, batch_size=16, 
                          sampler=sr_sampler_train, collate_fn=custom_collate_fn)
test_loader = DataLoader(dataset, batch_size=16, 
                         sampler=sr_sampler_test, collate_fn=custom_collate_fn)

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Check if GPU is available and move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

correct = 0
total = 0
total_label = []
total_pred = []
total_outputs = []
label_lst = list(class_id_cnt.keys())
with torch.no_grad():
    for data in test_loader:
        images = data['image'].to(device)
        labels = data['class_id'].to(device)
        # if args.task == "action":
        #     labels_lst_form = labels.cpu().numpy().tolist()
        #     labels_new = [class_id_transform_dict[id_label] 
        #                   for id_label in labels_lst_form]
        #     labels = torch.tensor(labels_new).to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        total_label.extend(labels.cpu().tolist())
        total_pred.extend(predicted.cpu().tolist())
        total_outputs.extend(outputs.cpu().tolist())

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


print("Before Fine-tuning")
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

criterion = nn.CrossEntropyLoss()
BCE_criterion = nn.BCELoss()
l1_criterion = nn.L1Loss(reduction="none")
optimizer = optim.Adam(model.parameters(), lr=0.001)

grad_cam = GradCam(model=model, feature_module=model.layer4, target_layer_names=["1"], use_cuda=True)

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
        temp_size = (visual_exp_trans != 0).float()
        eff_loss = torch.sum(temp_loss * temp_size) / torch.sum(temp_size)
        trans_att_loss = torch.relu(torch.mean(eff_loss) - 0)

        '''att_map_labels_trans = torch.stack(visual_exp_trans)'''
        tempD = F.l1_loss(att_map, visual_exp_trans)
        trans_att_loss = trans_att_loss + tempD
    
    elif trans_type == "HAICS":
        temp_att_loss = BCE_criterion(att_map, visual_exp_trans * (visual_exp_trans > 0).float())
        mask = (visual_exp_trans != 0).float()
        trans_att_loss = torch.mean(temp_att_loss * mask)
    elif trans_type == "GRADIA":
        temp_att_loss = l1_criterion(att_map, visual_exp_trans * (visual_exp_trans > 0).float())
        trans_att_loss = torch.mean(temp_att_loss)
    return trans_att_loss

# Training Loop
for epoch in range(args.num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    running_pred_loss = 0.0
    running_att_loss = 0.0
        
    for i, data in enumerate(train_loader):
        inputs = data['image'].to(device)
        labels = data['class_id'].to(device)
        # if args.task == "action":
        #     labels_lst_form = labels.cpu().numpy().tolist()
        #     labels_new = [class_id_transform_dict[id_label] 
        #                   for id_label in labels_lst_form]
        #     labels = torch.tensor(labels_new).to(device)
        visual_exps = data['visual_exp'].to(device)
        loss_each = []
        for image, label, visual_exp in zip(inputs, labels, visual_exps):
            att_map, _ = grad_cam.get_attention_map(torch.unsqueeze(image, 0), label, norm='ReLU') # att_map: torch.Size([8, 8])
            if visual_exp.cpu().numpy().any():
                visual_exp_resized = normalize_and_resize(visual_exp) 
                l1_loss = F.l1_loss(att_map, visual_exp_resized, reduction='mean') # Original att loss
                # Transformed Att Loss
                visual_exp_resized_trans = visual_exp_transform(visual_exp_resized, "HAICS")
                l1_loss = l1_loss + cal_trans_att_loss(att_map, visual_exp_resized_trans, "HAICS")
                loss_each.append(l1_loss)
        if len(loss_each):        
            att_loss = torch.stack(loss_each).sum() 
        else: 
            att_loss = torch.zeros(1, device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float32)  # Adjust device as needed
          
        optimizer.zero_grad()
                      
        outputs = model(inputs)
        pred_loss = criterion(outputs, labels)
        loss = pred_loss + args.att_weight * att_loss

        loss.backward()
        optimizer.step()
          
        # Update running totals
        running_loss += loss.item()
        running_pred_loss += pred_loss.item()
        running_att_loss += att_loss.item()

        # Print averaged losses every 100 batches
        if (i + 1) % 20 == 0:
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}, '
                  f'pred_loss: {running_pred_loss / 100:.3f}, '
                  f'att_loss: {running_att_loss / 100:.3f}')
            running_loss = 0.0
            running_pred_loss = 0.0
            running_att_loss = 0.0

    print('Finished Training')

# Testing Loop
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0
total_label_tuned = []
total_pred_tuned = []
total_outputs_tuned = []
with torch.no_grad():
    for data in test_loader:
        images = data['image'].to(device)
        labels = data['class_id'].to(device)
        # if args.task == "action":
        #     labels_lst_form = labels.cpu().numpy().tolist()
        #     labels_new = [class_id_transform_dict[id_label] 
        #                   for id_label in labels_lst_form]
        #     labels = torch.tensor(labels_new).to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        total_label_tuned.extend(labels.cpu().tolist())
        total_pred_tuned.extend(predicted.cpu().tolist())
        total_outputs_tuned.extend(outputs.cpu().tolist())

total_outputs_tuned = np.array(total_outputs_tuned)
total_label_tuned = np.array(total_label_tuned)

test_accuracy_tuned = accuracy_score(total_label_tuned, total_pred_tuned)
test_recall_tuned_micro = recall_score(total_label_tuned, total_pred_tuned, average="micro")
test_precision_tuned_micro = precision_score(total_label_tuned, total_pred_tuned, average="micro")
test_f1_tuned_micro = f1_score(total_label_tuned, total_pred_tuned, average="micro")

soft_max_outputs_tuned = torch.tensor(total_outputs_tuned)
soft_max_outputs_tuned = F.softmax(soft_max_outputs_tuned)

test_auc_tuned_micro = roc_auc_score(total_label_tuned, soft_max_outputs_tuned.numpy(), 
                                     average="micro", multi_class="ovr")

test_recall_tuned_macro = recall_score(total_label_tuned, total_pred_tuned, average="macro")
test_precision_tuned_macro = precision_score(total_label_tuned, total_pred_tuned, average="macro")
test_f1_tuned_macro = f1_score(total_label_tuned, total_pred_tuned, average="macro")
test_auc_tuned_macro = roc_auc_score(total_label_tuned, soft_max_outputs_tuned.numpy(), 
                                     average="macro", multi_class="ovr")

print("After Fine-tuning")
tb = pt.PrettyTable()
tb.field_names = ["", "Accuracy", "Recall", "Precision", "F1", "AUC"]
tb.add_row(
    ["Micro",test_accuracy_tuned, test_recall_tuned_micro, 
     test_precision_tuned_micro, test_f1_tuned_micro, test_auc_tuned_micro]
)
tb.add_row(
    ['Macro', test_accuracy_tuned, test_recall_tuned_macro, 
     test_precision_tuned_macro, test_f1_tuned_macro, test_auc_tuned_macro]
)
print(tb)