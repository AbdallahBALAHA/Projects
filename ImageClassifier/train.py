import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from collections import OrderedDict
from torchvision import datasets, transforms, models
import numpy as np
import json
import argparse

parser = argparse.ArgumentParser(description="Parser for a deep learning training script")

parser.add_argument('data_dir', help='Mandatory argument: Provide the directory containing the training data.', type=str)
parser.add_argument('--save_dir', help='Optional argument: Specify the directory to save the trained model.', type=str)
parser.add_argument('--arch', help='Optional argument: Specify the neural network architecture, you are able to use alexnet. If not specified, densenet121 will be used by default.', type=str)
parser.add_argument('--lrn', help='Optional argument: Set the initial learning rate. Default value is 0.002.', type=float)
parser.add_argument('--hidden_units', help='Optional argument: Number of hidden units in the classifier. Default value is 2048.', type=int)
parser.add_argument('--epochs', help='Mandatory argument: Maximum number of training epochs. Early stop when valid-accuracy >= 0.90', type=int)
parser.add_argument('--GPU', help='Optional argument: Specify to use GPU for training.', type=str)

args = parser.parse_args()


data_dir = args.data_dir
device = 'cuda' if args.GPU == 'GPU' else 'cpu'
learn_rate = args.lrn if args.lrn else 0.002
epoch_no = args.epochs if args.epochs else 10
arch = 'alexnet' if args.arch == 'alexnet' else 'densenet121'
hidden_units = args.hidden_units
save_dir = args.save_dir if args.save_dir else 'checkpoint_1.pth'

if data_dir:
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(224),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=True)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


def train_valid_save_model(arch, hidden_units, device, train_dataset, trainloader, validloader, learn_rate, epoch_no, save_dir):
    print(arch)
    if arch == 'alexnet':
        model = models.alexnet(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False
        if hidden_units:
            classifier = nn.Sequential(OrderedDict([
                                    ('fc1', nn.Linear (9216, 4096)),
                                    ('relu', nn.ReLU ()),
                                    ('dropout', nn.Dropout (p = 0.2)),
                                    ('fc2', nn.Linear (4096, hidden_units)),
                                    ('relu', nn.ReLU ()),
                                    ('dropout', nn.Dropout (p = 0.2)),
                                    ('fc3', nn.Linear(hidden_units, 102)),
                                    ('output', nn.LogSoftmax(dim=1))
                                    ]))
        else:
            classifier = nn.Sequential(OrderedDict([
                                    ('fc1', nn.Linear (9216, 4096)),
                                    ('relu', nn.ReLU ()),
                                    ('dropout', nn.Dropout (p = 0.2)),
                                    ('fc2', nn.Linear (4096, 2048)),
                                    ('relu', nn.ReLU ()),
                                    ('dropout', nn.Dropout (p = 0.2)),
                                    ('fc3', nn.Linear(2048, 102)),
                                    ('output', nn.LogSoftmax(dim=1))
                                    ]))
    else:
        model = models.densenet121(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        if hidden_units:
            classifier = nn.Sequential(OrderedDict([
                                    ('fc1', nn.Linear (1024, 512)),
                                    ('relu', nn.ReLU ()),
                                    ('dropout', nn.Dropout (p = 0.2)),
                                    ('fc2', nn.Linear (512, hidden_units)),
                                    ('relu', nn.ReLU ()),
                                    ('dropout', nn.Dropout (p = 0.2)),
                                    ('fc3', nn.Linear(hidden_units, 102)),
                                    ('output', nn.LogSoftmax(dim=1))
                                    ]))
        else:
            classifier = nn.Sequential(OrderedDict([
                                    ('fc1', nn.Linear (1024, 512)),
                                    ('relu', nn.ReLU ()),
                                    ('dropout', nn.Dropout (p = 0.2)),
                                    ('fc2', nn.Linear (512, 256)),
                                    ('relu', nn.ReLU ()),
                                    ('dropout', nn.Dropout (p = 0.2)),
                                    ('fc3', nn.Linear(256, 102)),
                                    ('output', nn.LogSoftmax(dim=1))
                                    ]))           

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.85)

    model.to(device);

    epochs = epoch_no
    steps = 0
    running_loss = 0
    print_every = 50

    for epoch in range(epochs):
        print(arch)
        scheduler.step()
        print("Learning Rate:", scheduler.get_lr())
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                    f"Valid accuracy: {accuracy/len(validloader):.3f}")

                running_loss = 0
                model.train()

        if accuracy/len(validloader) >= 0.90:

            print(f"Valid accuracy: {accuracy/len(validloader):.3f}")
            print("Done")
            break
            
    model.to("cpu")
    model.class_to_idx = train_dataset.class_to_idx

    chck = {
                'classifier' : model.classifier,
                'map_to_class' : model.class_to_idx,
                'state_dict' : model.state_dict,
                'arch' : arch
    }

    torch.save(chck, save_dir)

train_valid_save_model(arch, hidden_units, device, train_dataset, trainloader, validloader, learn_rate, epoch_no, save_dir)