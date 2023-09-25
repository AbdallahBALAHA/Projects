import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from collections import OrderedDict
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse

parser = argparse.ArgumentParser(description="Script for Predicting Image Categories")

parser.add_argument('image_dir', help='Mandatory argument: Provide the path to the image.', type=str)
parser.add_argument('load_dir', help='Mandatory argument: Provide the path to the checkpoint.', type=str)
parser.add_argument('--top_k', help='Optional: Number of top K most likely classes to display.', type=int)
parser.add_argument('--category_names', help='Optional: JSON file containing the mapping of categories to real names.', type=str)
parser.add_argument('--GPU', help='Optional: Use GPU for prediction.', type=str)

args = parser.parse_args()

image_dir = args.image_dir
load_dir = args.load_dir
top_k = args.top_k if args.top_k else 5
category_names = args.category_names if args.category_names else 'cat_to_name.json'
device = 'cuda' if args.GPU == 'GPU' else 'cpu'


def load_model(path):
    chck = torch.load(path)

    if chck['arch'] == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        model = models.densenet121(pretrained=True)
    
    model.classifier = chck['classifier']
    model.state_dict = chck['state_dict']
    model.class_to_idx = chck['map_to_class']
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image).convert("RGB")
    transform = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    torch_im = transform(im)
    np_im = torch_im.numpy()
    
    return np_im



def predict(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Loading & preprocessing
    image = process_image(image_path)
    if device == 'cuda':
        image_tensor = torch.from_numpy (image).type (torch.cuda.FloatTensor)
    else:
        image_tensor = torch.from_numpy (image).type (torch.FloatTensor)
        
    image_tensor = image_tensor.unsqueeze(dim=0)

    model.to(device)
    image_tensor.to(device)

    with torch.no_grad():
        output = model.forward(image_tensor)

    probs = torch.exp(output)
    top_probs, top_idxs = probs.topk(topk)
    top_probs = top_probs.cpu()
    top_idxs = top_idxs.cpu()
    top_probs = top_probs.numpy()[0]
    top_idxs = top_idxs.numpy()[0]

    # Map idx 2 labels
    class_to_idx = model.class_to_idx
    idx_to_class = {val: key for key, val in class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_idxs]

    return top_probs, top_classes


model = load_model(load_dir)

with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

img = process_image(image_dir)

probs, classes = predict(image_dir, model, top_k, device)

class_names = [cat_to_name[item] for item in classes]

for i in range(len(class_names)):
    print('Name: ', class_names[i], '\nProbability: ', probs[i])