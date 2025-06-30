# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:00:50 2024

"""
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import numpy as np
from torch.utils.data import Subset, DataLoader
import torch.optim as optim
import train_pipeline as training
import matplotlib.pyplot as plt

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # Ensure reproducibility for certain operations
    torch.backends.cudnn.benchmark = False 
set_seed()
"""
Data pre-processing
"""

# Data preparation
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # ResNet18 typically takes 224x224 images
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

total_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
subset_fraction = 0.25
num_samples = int(len(total_dataset) * subset_fraction)
indices = np.random.choice(len(total_dataset), num_samples, replace=False)

train_subset = Subset(total_dataset, indices)
train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)


data_loader = torch.utils.data.DataLoader(train_subset, batch_size=len(train_subset), shuffle=False)
eval_loader = torch.utils.data.DataLoader(train_subset, batch_size = 1000, shuffle = False)
data_iter = iter(data_loader)
train_data, train_targets = next(data_iter)  # Load the entire dataset into memory

def initialize_weights_kaiming_uniform(model):
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
def compute_initial_loss(model, data_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    with torch.no_grad():  # Disable gradient computation
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, targets)  # Compute loss
            total_loss += loss.item() * inputs.size(0)  # Weighted by batch size
    return total_loss / len(data_loader.dataset) 


"""
Initialize Model: ResNet18
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=False)  # ResNet18 model
model.fc = nn.Linear(512, 10)  # CIFAR-10 has 10 classes
initialize_weights_kaiming_uniform(model)
torch.save(model.state_dict(), "kaiming_intialization.pth")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
train_pip = training.train_pip(model, criterion)
# Define the model, criterion
train_pip.batch_size = 32
train_pip.independent_batch = 32
train_pip.num_epochs = 20
train_pip.clip_threshold = 5.0
train_pip.delta = 1e-1

training_pipelines = {
    "SGD": (train_pip.SGD_train, {"lr": 1e-5}),
    "Adam": (train_pip.Adam_train, {"lr": 1e-4}),
    "Adagrad": (train_pip.Adagrad_train, {"lr": 1e-3}),
    "NSGD": (train_pip.NSGD_train, {"lr": 5e-2}),
    "NSGDm": (train_pip.NSGDm_train, {"lr": 1e-2}),
    "Clip_SGD": (train_pip.ClipSGD_train, {"lr": 8e-2}),
    "IANSGD": (train_pip.INSGD_train, {"lr": 8e-2}),
    
}
all_loss_curves = {}
lgd_size = 20
num_size = 18
label_size = 18
for (name, (train_function,params)) in training_pipelines.items():
    print(f"Running training pipeline: {name}")
    
    # Initialize a new ResNet18 model
    
    # Apply Kaiming uniform initialization
    model.load_state_dict(torch.load("kaiming_intialization.pth"))
    train_pip.lr = params["lr"]
    # Train the model using the pipeline
    initial_loss = compute_initial_loss(model, eval_loader, criterion, device)
    update_model, train_loss_list = train_function(train_data, train_targets)
    all_loss_curves[name] = [initial_loss]+train_loss_list

plt.rc('axes', labelsize=label_size)   # fontsize of the x and y labels
plt.rc('xtick', labelsize=num_size)    # fontsize of the tick labels
plt.rc('ytick', labelsize=num_size)    # fontsize of the tick labels
plt.figure(figsize=(8, 6))
num_epochs = train_pip.num_epochs
#markers = ['o-','v-','*-','+-','x-','s-','p-']
for idx,(name, loss_curve) in enumerate(all_loss_curves.items()):
    #mark = markers[idx]
    plt.plot(range(0, num_epochs + 1),loss_curve,linewidth=2.5,label=name)
plt.ylim(0, 40)
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.15)
plt.legend(prop={'size':lgd_size},loc='upper right',ncol = 1)
plt.xlabel("Epochs")
plt.ylabel("Loss")
#plt.title("Training Loss Curves for ResNet18")
plt.grid(True)
plt.savefig("ResNet18.png")
plt.show()

# Train the model using I-NSGD
#trained_model, train_loss_list = train_pip.INSGD_train(all_data, all_targets)

# Save the trained model
#torch.save(trained_model.state_dict(), 'resnet18_insgd.pth')
