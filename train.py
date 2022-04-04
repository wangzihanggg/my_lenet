import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=36, shuffle=True, num_workers=0)

test_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
test_loader = torch.utils.data.DataLoader(train_set, batch_size=10000, shuffle=True, num_workers=0)

test_data_iter = iter(test_loader)
test_image, test_label = test_data_iter.next()

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
