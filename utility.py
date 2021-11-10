import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms as T
import torchvision.models as models
import json

from collections import OrderedDict
from PIL import Image

# Train Process
def load_process(data_dir='./'):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define transforms
    # TODO: Define your transforms for the training, validation, and testing sets
    
    train_transforms = T.Compose([
        T.RandomRotation(60),
        T.RandomHorizontalFlip(),
        T.RandomResizedCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transforms = T.Compose([
        T.Resize(255),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_transforms = T.Compose([
        T.Resize(255),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # TODO: Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(data_dir + '/train',
                                          transform=train_transforms)
    test_datasets = datasets.ImageFolder(data_dir + '/test',
                                         transform=test_transforms)
    valid_datasets = datasets.ImageFolder(data_dir + '/valid',
                                          transform=valid_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size=64)
    validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=64)

    print("<========= DONE LOADING DATA ==========>")

    return trainloader, testloader, validloader, train_datasets