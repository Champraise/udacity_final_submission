import argparse
import numpy as np
import json

import torch
from torchvision import datasets, transforms, models

import argparse
import my_lib as lib

def argparser():
    parser = argparse.ArgumentParser(description="Predict Image Class")
    
    parser.add_argument('image_path', 
                    default='flowers/test/1/image_06752.jpg',
                    action="store", 
                    type=str, 
                    help="Define the directory for the Image")
    
    parser.add_argument('checkpoint_pth', 
                    default='checkpoint.pth',
                    action="store", 
                    type=str, 
                    help="Define the directory to the PTH file")
    
    parser.add_argument('--top_k', 
                    default=5, 
                    dest="top_k", 
                    action="store", 
                    type=int,
                    help="To show the top_k Prediction")
    parser.add_argument('--category_names', 
                    dest="category_names", 
                    action="store", 
                    default='cat_to_name.json',
                    help="Define The category name")
    parser.add_argument('--device', 
                        type=str, 
                        default='cuda', 
                        choices=['cuda', 'cpu'],
                        help='set device mode cuda or cpu (default=cuda)')
    args = parser.parse_args()
    
    return args
# Get Keyword Args for Prediction
args = argparser()
image_path = args.image_path
checkpoint_pth = args.checkpoint_pth
top_k = args.top_k
category_names = args.category_names

def main():
    
    # Get Keyword Args for Prediction
    args = argparser()
    
    # Defining the device 
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print('CUDA GPU mode is enabled now')
    elif args.device == 'cpu':
        device = torch.device('cpu')
        print('CPU mode is enabled now')
    else:
        print('CUDA GPU is not supported on this system so switching to CPU mode')
        device = torch.device('cpu')


    # Open the json file. 
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    model, checkpoint = lib.load_model(checkpoint_pth)
    probs, class_indices = lib.predict(checkpoint, image_path, model, device, top_k)

    # Mapping the predicited names to predicited class
    predicted_flower_names = [cat_to_name[idx] for idx in class_indices]

    print("{} most likely predicted flower names = {}".format(top_k, predicted_flower_names))
    print("{} most likely predicted probability = {}".format(top_k, probs))
    print("The model predicted flower name as {} with {:.3f}% probability".format(predicted_flower_names[0], 100 * probs[0]))
    
if __name__ == '__main__': main()