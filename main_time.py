import os
import numpy as np
import glob
import PIL.Image as Image
from tqdm import tqdm
import cv2 as cv
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
import wandb
wandb.login(key='4aaf96e30165bfe476963bc860d96770512c8060')

from helpers import parse_xml, get_proposals, _EdgeBox, _SelectiveSearch, cut_patches
from typing import Union
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix


# Set device
print("The code will run on GPU." if torch.cuda.is_available() else "The code will run on CPU.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    os.chdir("./IDLCV_OD")
except:
    try:
        os.chdir("./Courses/IDLCV_OD")
    except:
        ...

random.seed(6284)
np.random.seed(6284)
torch.manual_seed(6284)


# ############################# #
# Hyperparameters IMPORTANT!!!! #
# ############################# #

search_method:Union["SS", "EB"] = "SS"
generator = _SelectiveSearch if search_method == "SS" else _EdgeBox
k1 = 0.2
k2 = 0.6

print(f"search_method: {search_method}, k1: {k1}, k2: {k2}")

# ######## #
# Datasets #
# ######## #

root = "./Data/Potholes"
ids_train = sorted(set([fn.split("-")[1].split(".")[0] for fn in os.listdir(f"{root}/train")]), key =lambda x: int(x))
ids_test = sorted(set([fn.split("-")[1].split(".")[0] for fn in os.listdir(f"{root}/test")]), key =lambda x: int(x))
ids_validation = sorted(set([fn.split("-")[1].split(".")[0] for fn in os.listdir(f"{root}/validation")]), key =lambda x: int(x))

# Train
image_paths_train = [f"{root}/train/img-{id}.jpg" for id in ids_train]
GT_train = [parse_xml(f"{root}/train/img-{id}.xml")[1] for id in ids_train]
pos_proposals_train = []
pos_proposals_train_positions = []
neg_proposals_train = []
neg_proposals_train_positions = []
for ip in tqdm(image_paths_train):
    image = cv.imread(ip)
    pp, _np = get_proposals(image, GT_train, k1=k1, k2=k2, generator=generator)
    pos_proposals_train_positions += pp
    neg_proposals_train_positions += _np
    # Cut into patches
    pp, _np = cut_patches(image, pp, _np)
    pos_proposals_train += pp
    neg_proposals_train += _np

# Test
image_paths_test = [f"{root}/test/img-{id}.jpg" for id in ids_test]
GT_test = [parse_xml(f"{root}/test/img-{id}.xml")[1] for id in ids_test]
pos_proposals_test = []
pos_proposals_test_positions = []
neg_proposals_test = []
neg_proposals_test_positions = []
for ip in tqdm(image_paths_test):
    image = cv.imread(ip)
    pp, _np = get_proposals(image, GT_test, k1=k1, k2=k2, generator=generator)
    pos_proposals_test_positions += pp
    neg_proposals_test_positions += _np
    # Cut into patches
    pp, _np = cut_patches(image, pp, _np)
    pos_proposals_test += pp
    neg_proposals_test += _np

# Validation
image_paths_validation = [f"{root}/validation/img-{id}.jpg" for id in ids_validation]
GT_validation = [parse_xml(f"{root}/validation/img-{id}.xml")[1] for id in ids_validation]
pos_proposals_validation = []
pos_proposals_validation_positions = []
neg_proposals_validation = []
neg_proposals_validation_positions = []
for ip in tqdm(image_paths_validation):
    image = cv.imread(ip)
    pp, _np = get_proposals(image, GT_validation, k1=k1, k2=k2, generator=generator)
    pos_proposals_validation_positions += pp
    neg_proposals_validation_positions += _np
    # Cut into patches
    pp, _np = cut_patches(image, pp, _np)
    pos_proposals_validation += pp
    neg_proposals_validation += _np


# Define the CustomDataset class
class CustomDataset(Dataset):
    def __init__(self, split:Union["train", "test", "validation"]='train', patch_transform:transforms.Compose=None, prop_pos:float=0.25):
        self.prop_pos = prop_pos
        self.patch_transform = patch_transform
        self.pos_proposals = {"train": pos_proposals_train, "test": pos_proposals_test, "validation": pos_proposals_validation}[split]
        self.neg_proposals = {"train": neg_proposals_train, "test": neg_proposals_test, "validation": neg_proposals_validation}[split]
        print(f'# Positive Proposals:{len(self.pos_proposals)}')
        print(f'# Negative Proposals:{len(self.neg_proposals)}')


    def __len__(self):
        return len(self.pos_proposals) + len(self.neg_proposals)
    
    def __getitem__(self, idx):

        # Get positive class
        target = 1 # 1 is class, 0 is background
        proposal = None
        index = None
        if random.random() < self.prop_pos:
            index = idx % len(self.pos_proposals)
            proposal = self.pos_proposals[index]
        else:
            target = 0
            index = idx % len(self.neg_proposals)
            proposal = self.neg_proposals[index]

        # Apply transformations
        if self.patch_transform:
            proposal = self.patch_transform(Image.fromarray(proposal))

        return proposal, target, index ####             TEITUR HERE IS THE INDEX TO GET WHERE THE PATCH IS - USE: (TARGET + SPLIT TYPE + INDEX) TO GET E.G neg_proposals_validation_positions[index], see code above custom dataset class


# Define helper functions
def unnormalize(tensor):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

# ############## #
# Some set stuff #
# ############## #
SHUFFLE = True
NUM_WORKERS = 1

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Update build_datasets function
def build_datasets(config):

    # The transform
    patch_transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # The datasets
    train_dataset = CustomDataset(split='train', patch_transform=patch_transform, prop_pos=config.prop_pos)
    val_dataset = CustomDataset(split='validation', patch_transform=patch_transform, prop_pos=config.prop_pos)
    test_dataset = CustomDataset(split='test', patch_transform=patch_transform, prop_pos=config.prop_pos)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=SHUFFLE, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=NUM_WORKERS)

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


# ######### #
# Visualize #
# ######### #
# Plot images, predictions and their masks - to show dataset
def visualizer(dataloader, model, num_samples=16, title="Final_Validation_Visualization", run_id="vis"):
    """Visualize predictions on the validation set and save them with the given run_id."""
    model.eval()
    plt.figure(figsize=(10, 20))
    os.makedirs(f"./vis/vis/{run_id}", exist_ok=True)

    with torch.no_grad():
        for i, (image, target, _) in enumerate(dataloader):
            if i >= num_samples: break

            image = image.to(device)
            pred = torch.sigmoid(model(image)).cpu().squeeze(0)[0].item()

            # Plot the image
            plt.subplot(4, 4, i + 1)
            plt.imshow(transforms.ToPILImage()(unnormalize(image.squeeze(0))))
            plt.title(f"Target: {target[0].item()}, Pred: {round(pred, 3)}")
            plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"./vis/vis/{run_id}/{title}.png")
    plt.close()

    
# Define the build_optimizer function
def build_optimizer(model, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    return optimizer

# Define loss functions
def bce_loss(y_pred, y_real):
    return F.binary_cross_entropy_with_logits(y_pred, y_real)

# Define the compute_metrics function

def compute_metrics(preds, targets):

    preds[preds < 0.5] = 0 
    preds[preds >= 0.5] = 1

    # Convert to integer type
    preds = preds.astype(int)
    targets = targets.astype(int)

    # Calculate accuracy
    acc = accuracy_score(targets, preds)
    
    # Calculate sensitivity (recall)
    sensitivity = recall_score(targets, preds)
    
    # Calculate specificity using the confusion matrix
    tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
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
        for data, target, _ in train_loader:
            data, target = data.to(device), target.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            train_preds.append(torch.sigmoid(output).cpu())
            train_targets.append(target.long().cpu())

        # Compute training metrics
        train_acc, train_sensitivity, train_specificity = compute_metrics(torch.concat(train_preds).detach().numpy(), torch.concat(train_targets).numpy())

        # Validation phase
        val_loss = []
        val_preds = []
        val_targets = []
        model.eval()

        with torch.no_grad():
            for data, target, _ in val_loader:
                data, target = data.to(device), target.to(device).float().unsqueeze(1)
                output = model(data)
                loss = criterion(output, target)

                val_loss.append(loss.item())
                val_preds.append(torch.sigmoid(output).cpu())
                val_targets.append(target.long().cpu())

        # Compute validation metrics
        val_acc, val_sensitivity, val_specificity = compute_metrics(torch.cat(val_preds).numpy(), torch.cat(val_targets).numpy())

        # Test phase
        visualizer(test_loader, model, 16, f"{epoch}-test-vis", run_id)

        test_preds = []
        test_targets = []
        model.eval()

        with torch.no_grad():
            for data, target, _ in test_loader:
                data, target = data.to(device), target.to(device).float().unsqueeze(1)
                output = model(data)

                test_preds.append(torch.sigmoid(output).cpu())
                test_targets.append(target.long().cpu())

        # Compute validation metrics
        test_acc, test_sensitivity, test_specificity = compute_metrics(torch.cat(test_preds).numpy(), torch.cat(test_targets).numpy())

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


sweep_config = {
    'method' : 'bayes'#'random', #'bayes' "grid"
}

metric = {
    'name': 'loss',
    'goal': 'minimize'
  }

parameters_dict = {
    'prop_pos': {
        'values': [0.25, 0.5] 
    },
    'dropout' : {
        'values' : [0.2, 0.5]
    },
    'epochs' : {
        'value': 300
    },
    'learning_rate' : {
        "values": [0.0001, 0.001, 0.01]
    },
    'batch_size': {
        "values": [16, 32]
    },
    'image_size':{
        'value': 128
    },
    "num_layers": {
        "values": list(range(4, 7))
    }
}
sweep_config['metric'] = metric
sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project='Object_Detection')

class Base_Network(nn.Module):
    def __init__(self, dropout, num_layers, feature_size=128, base_channel_sz = 8):
        super(Base_Network, self).__init__()

        self.beginning = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=base_channel_sz, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(base_channel_sz),
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.MaxPool2d(kernel_size=2, stride=2)

        )
        feature_size //= 2 # Convolution -> Padding -> Pooling

        layers = []
        for i in range(num_layers):
          layers.append(nn.Conv2d(in_channels=base_channel_sz*(i+1), out_channels=base_channel_sz*(i+2), kernel_size=3, padding=1))
          layers.append(nn.BatchNorm2d(base_channel_sz*(i+2)))
          layers.append(nn.ReLU())
          layers.append(nn.Dropout(dropout))
          
          layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
          feature_size //= 2 # Convolution -> Padding -> Pooling

        self.convolutional = nn.Sequential(*layers)
        self.fully_connected = nn.Sequential(
            nn.Linear(in_features= base_channel_sz * (num_layers + 1) * feature_size * feature_size, out_features=512),
            nn.ReLU(),
            nn.Dropout(dropout), # added dropout to reduce overfitting

            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.beginning(x)
        x = self.convolutional(x)
        #reshape x so it becomes flat, except for the first dimension (which is the minibatch)
        x = x.view(x.size(0), -1)

        x = self.fully_connected(x)
        return x
    
def run_wandb(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        config = wandb.config

        # Include the run id so that we can identify the saved models
        run_id = wandb.run.id 
        config.run_id = run_id
        wandb.run.name = f"Run {run_id}"

        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = build_datasets(config)
        model = Base_Network(config.dropout, config.num_layers) #build_network(config.fc_layer_size, config.dropout)
        model.to(device)
        print(model)

        optimizer = build_optimizer(model, config.learning_rate)

        # Generate a random id for this run and this model
        # train(model, optimizer, train_loader, val_loader, train_dataset, val_dataset, config.epochs, run_id)

        train(model, optimizer, train_loader, val_loader, test_loader, criterion=bce_loss, num_epochs= config.epochs, run_id=run_id)

wandb.agent(sweep_id, run_wandb)