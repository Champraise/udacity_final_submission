import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms as T
import torchvision.models as models
import json
import utility as ut

from collections import OrderedDict
from PIL import Image

# Build Model 
def build_model(arch, hidden_units, dropout, learning_rate, device):
    
    # Load the json file for mapping the name labels 
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
        
    # Calculate number of desired output
    output_layer = len(cat_to_name)

    # Use available pytorch model
    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
        # Freeze parameters so we don't backpropagate through them
        for param in model.parameters():
            param.requires_grad = False
        
        # Classifier for vgg16.
        classifier = nn.Sequential(OrderedDict([
                 ('fc1', nn.Linear(25088, hidden_units)),
                 ('relu', nn.ReLU()), 
                 ('dropout', nn.Dropout(dropout)),
                 ('fc2', nn.Linear(hidden_units, output_layer)),
                 ('output', nn.LogSoftmax(dim=1))]))
    
    # Definition for model densenet121:
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
            
        # Freeze parameters so we don't backpropagate through them
        for param in model.parameters():
            param.requires_grad = False
        
        # Classifier for vgg16
        classifier = nn.Sequential(OrderedDict([
                 ('fc1', nn.Linear(1024, hidden_units)),
                 ('relu', nn.ReLU()), 
                 ('dropout', nn.Dropout(dropout)),
                 ('fc2', nn.Linear(hidden_units, output_layer)),
                 ('output', nn.LogSoftmax(dim=1))]))
    else:
        print("Im sorry but {} is not a valid model. You use vgg16 or densenet121?".format(arch))

    # Stating the classifier
    model.classifier = classifier
    
    # Define Loss
    criterion = nn.NLLLoss()
    
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

    model.to(device)
    
    print("Our model: \n\n", model, '\n')
    print("The state dict keys: \n\n", model.state_dict().keys())
    print("<============ Done Building Model ===========>")
    
    return model, criterion, optimizer

#Train the class
def train_model(model, criterion, optimizer, trainloader, validloader, testloader, epochs, device, print_every = 20):
    steps = 0
    running_loss = 0
    train_losses, valid_losses = [], []
    print("<========== Strating to train ===========>")
    for e in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            
            # Use if GPU is available
            #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Move input and label tensor to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            logps = model(inputs)
            loss = criterion(logps, labels)

            # Clear the gradients, do this because gradients are accumulated
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)

                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                train_losses.append(running_loss / print_every)
                valid_losses.append(valid_loss / len(validloader))

                print(f"Epoch {e+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {100*(accuracy/len(validloader)):.3f}%")
                running_loss = 0

                # set model back to train mode
                model.train()
    print("<==========Training Done ==========>")

def save_checkpoint(arch, epochs, hidden_units, learning_rate, model, optimizer, train_datasets, save_dir):
    model.class_to_idx = train_datasets.class_to_idx
    checkpoint = {
        'architure': arch,
        'epochs': epochs,
        'hidden_units': hidden_units,
        'learning_rate': learning_rate,
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'optimizer': optimizer,
        'class_to_idx': model.class_to_idx,
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, save_dir)
    print("<========= Done Saving Checkpoint =========>")

def load_model(checkpoint_pth):
    print("<========== Loading Checkpoint ============>")
        
    checkpoint = torch.load(checkpoint_pth)

    if checkpoint['architure'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif checkpoint['architure'] == 'densenet121':
        model = models.densenet121(pretrained=True)
        
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']

    # Load state_dict
    model.load_state_dict(checkpoint['state_dict'])

    optimizer = checkpoint['optimizer']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print("<========= Done Loading Checkpoint =============>")
    return model, checkpoint['class_to_idx']


# Image process
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image_path)
    transforms = T.Compose([
        T.Resize(255),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transforms(img)

    return image

# Predict Function 
def predict(checkpoint, image_path, model, device, top_k):
    # TODO: Implement the code to predict the class from an image file
    # Define device selection process
    print("<====== start of predict func ========>")
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move model to device
    model.to(device)
    
    # Image processing
    image = process_image(image_path)
    
    # Move image to device
    image = image.to(device)
    
    # values of class indices
    img_class_dict = {value: key for key, value in model.class_to_idx.items()}

    model.eval()

    # Turn off the gradients
    with torch.no_grad():
        image.unsqueeze_(0)
        # Forward process
        prediction = model.forward(image)
        ps = torch.exp(prediction)
        probs, classes = ps.topk(top_k)
        probs, classes = probs[0].tolist(), classes[0].tolist()

        class_indices = []
        for c in classes:
            class_indices.append(img_class_dict[c])
    print("<=========end of predict =========>")
    return probs, class_indices