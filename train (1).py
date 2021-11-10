import argparse
import utility as ut
import my_lib as lib


import torch
from torchvision import datasets, transforms, models

# Create the parser
parser = argparse.ArgumentParser(description='Training File for a new Network on a dataset')

# Add the default command to paser
parser.add_argument('data_dir',
                    action="store",
                    default="./flowers/",
                    help="Define the directory for data ")

# Add the option and positional arguments to the parser
parser.add_argument('--arch',
                    dest="arch",
                    type=str,
                    action="store",
                    default="vgg16",
                    help="Prefered model in torch vision")

parser.add_argument("-lr", '--learning_rate',
                    dest="learning_rate",
                    type=float,
                    action="store",
                    default=0.003,
                    help="Learning rate. Default = 0.003")

parser.add_argument("--hidden_units",
                    dest="hidden_units",
                    type=int,
                    default= 1000, 
                    help="Number of node in hidden layer")

parser.add_argument('--epochs',
                    dest="epochs",
                    action="store",
                    type=int,
                    default=3,
                    help="Number of epochs. Default = 3")

parser.add_argument("--dropout",
                    dest="dropout",
                    type=float,
                    default=0.03,
                    help="Value of dropout")

parser.add_argument('--save_dir', 
                    type=str, 
                    default='checkpoint.pth',
                    help='set path of the model checkpoint (default=checkpoint.pth)')

parser.add_argument('--device', 
                    type=str, 
                    default='cuda', 
                    choices=['cuda', 'cpu'],
                    help='set device mode cuda or cpu (default=cuda)')

# Excute the .parse_args()
args = parser.parse_args()


arch = args.arch
epochs = args.epochs
hidden_units = args.hidden_units
dropout = args.dropout
save_dir = args.save_dir
learning_rate = args.learning_rate

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

trainloader, validloader, testloader, train_datasets = ut.load_process(args.data_dir)
model, criterion, optimizer = lib.build_model(arch, learning_rate, hidden_units, dropout, device)
lib.train_model(model, criterion, optimizer, trainloader, validloader, testloader, epochs, device, 20)
lib.save_checkpoint(arch, epochs, model, optimizer, train_datasets, save_dir)
# Checkpoint saved sucessfully 