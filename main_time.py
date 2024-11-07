import os
import numpy as np
import glob
import PIL.Image as Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
import json

from helpers import parse_xml, get_proposals, _EdgeBox, _SelectiveSearch
from typing import Union

# Set device
print("The code will run on GPU." if torch.cuda.is_available() else "The code will run on CPU.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ######## #
# Datasets #
# ######## #

# Define the CustomDataset class
class CustomDataset(Dataset):
    def __init__(self, root="./Data/Potholes", split:Union["train", "test", "validation"]='train', patch_transform:transforms.Compose=None, prop_pos:float=0.25, search_method:Union["SS", "EB"]="SS", k1=0.3, k2=0.7):
        self.prop_pos = prop_pos
        self.patch_transform = patch_transform
        self.k1 = k1
        self.k2 = k2

        ids = set([fn.split("-")[1].split(".")[0] for fn in os.listdir(f"{root}/{split}")])
        self.image_paths = [f"{root}/{split}/img-{id}.jpg" for id in ids]
        self.GT = [parse_xml(f"{root}/{split}/img-{id}.xml")[1] for id in ids]
        self.generator = _SelectiveSearch if search_method == "SS" else _EdgeBox

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        
        # Get positive class
        target = 1 # 1 is class, 0 is background
        proposal = None
        if random.random() < self.prop_pos:
            proposal, _ = get_proposals(image, self.GT, num_pos_proposals=1, num_neg_proposals=0, k1=self.k1, k2=self.k2, generator=self.generator)
        else:
            target = 0
            _, proposal = get_proposals(image, self.GT, num_pos_proposals=0, num_neg_proposals=1, k1=self.k1, k2=self.k2, generator=self.generator)

        # Apply transformations
        if self.patch_transform:
            image = self.patch_transform(image)

        return image, target


# Define helper functions
def unnormalize(tensor, mean, std):
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

# ############## #
# Some set stuff #
# ############## #
SHUFFLE = True
NUM_WORKERS = 1

# Update build_datasets function
def build_datasets(config):

    # The transform
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    patch_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ColorJitter(),
        transforms.Normalize(mean=mean, std=std),
        transforms.Resize(config.image_size, config.image_size)
    ])

    # The datasets
    train_dataset = CustomDataset(split='train', patch_transform=patch_transform, prop_pos=config.prop_pos, search_method=config.search_method, k1=config.k1, k2=config.k2)
    val_dataset = CustomDataset(split='validation', patch_transform=patch_transform, prop_pos=config.prop_pos, search_method=config.search_method, k1=config.k1, k2=config.k2)
    test_dataset = CustomDataset(split='test', patch_transform=patch_transform, prop_pos=config.prop_pos, search_method=config.search_method, k1=config.k1, k2=config.k2)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=SHUFFLE, num_workers=NUM_WORKERS, prop_pos=config.prop_pos, search_method=config.search_method, k1=config.k1, k2=config.k2)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=NUM_WORKERS, prop_pos=config.prop_pos, search_method=config.search_method, k1=config.k1, k2=config.k2)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=NUM_WORKERS, prop_pos=config.prop_pos, search_method=config.search_method, k1=config.k1, k2=config.k2)

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset

# ######### #
# Visualize #
# ######### #
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
patch_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ColorJitter(),
    transforms.Normalize(mean=mean, std=std),
    transforms.Resize(64, 64)
])
test_dataset = CustomDataset(split='test', patch_transform=patch_transform, prop_pos=0.25, search_method="EB", k1=0.3, k2=0.7)

plt.figure(figsize=(20,20))
for i, (data, target) in enumerate(test_data):
    if i >= 16: break
    plt.subplot(4,4,i+1)
    plt.imshow(unnormalize(data))
    plt.title(['Background', 'Pot Hole'][target])
plt.savefig(f"./vis/arbitrary/test_dataset.png")
plt.close()

# Define the build_optimizer function
def build_optimizer(model, optimizer_name, learning_rate):
    if optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    return optimizer

# Define loss functions
def bce_loss(y_pred, y_real):
    return F.binary_cross_entropy_with_logits(y_pred, y_real)

# Define the compute_metrics function
def compute_metrics(preds, targets):
    acc = 0
    sensitivity = 0
    specificity = 0

    # TODO: calculate metrics

    return acc, sensitivity, specificity


# Update the train function
def train(model, optimizer, train_loader, val_loader, test_loader, criterion, num_epochs=10, run_id=""):
    out_dict = {
        'train_acc': [],
        'val_acc': [],
        "test_acc": [],
        'train_sensitivity': [],
        'val_sensitivity': [],
        "test_sensitivity": [],
        'train_specificity': [],
        'val_specificity': [],
        "test_specificity": [],
        'train_loss': [],
        'val_loss': [],
        "test_loss": []
    }

    for epoch in range(num_epochs):
        model.train()
        train_loss = []
        train_preds = []
        train_targets = []

        # Training phase
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            train_preds.append(torch.sigmoid(output).cpu())
            train_targets.append(target.long().cpu())

        # Compute training metrics
        train_dice, train_iou, train_acc, train_sensitivity, train_specificity = compute_metrics(train_preds, train_targets)

        # Validation phase
        val_loss = []
        val_preds = []
        val_targets = []
        model.eval()

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

                val_loss.append(loss.item())
                val_preds.append(torch.sigmoid(output).cpu())
                val_targets.append(target.long().cpu())

        # Compute validation metrics
        val_dice, val_iou, val_acc, val_sensitivity, val_specificity = compute_metrics(val_preds, val_targets)

        # Test phase
        test_preds = []
        test_targets = []
        model.eval()

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)

                test_preds.append(torch.sigmoid(output).cpu())
                test_targets.append(target.long().cpu())

        # Compute validation metrics
        test_acc, test_sensitivity, test_specificity = compute_metrics(test_preds, test_targets)

        # Record metrics
        out_dict['train_loss'].append(np.mean(train_loss))
        out_dict['val_loss'].append(np.mean(val_loss))

        out_dict['train_acc'].append(train_acc)
        out_dict['val_acc'].append(val_acc)
        out_dict['test_acc'].append(test_acc)

        out_dict['train_sensitivity'].append(train_sensitivity)
        out_dict['val_sensitivity'].append(val_sensitivity)
        out_dict['test_sensitivity'].append(test_sensitivity)

        out_dict['train_specificity'].append(train_specificity)
        out_dict['val_specificity'].append(val_specificity)
        out_dict['test_specificity'].append(test_specificity)

        # Log to WandB
        wandb.log({
            "val_acc": val_acc,
            "val_sensitivity": val_sensitivity,
            "val_specificity": val_specificity,
            "train_acc": train_acc,
            "train_sensitivity": train_sensitivity,
            "train_specificity": train_specificity,
            "test_acc": test_acc,
            "test_sensitivity": test_sensitivity,
            "test_specificity": test_specificity,
            "epoch": epoch,
            "run_id": run_id
        })

        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {out_dict['train_loss'][-1]:.3f}, "
              f"Val Loss: {out_dict['val_loss'][-1]:.3f}, "
              f"Train Acc: {train_acc*100:.2f}%, "
              f"Val Acc: {val_acc*100:.2f}%, "
              f"Test Acc: {test_acc*100:.2f}%")


