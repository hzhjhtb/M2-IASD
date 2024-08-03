#!/usr/bin/env python3 
import os
import argparse
import random
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import ssl
from FGSM import *
from PGD import *

ssl._create_default_https_context = ssl._create_unverified_context

use_cuda = torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")

valid_size = 1024 
batch_size = 32

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, img, labels):
        self.img = img
        self.labels = labels

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        return self.img[idx].squeeze(), self.labels[idx].squeeze()

############################------Basic neural network architecture (from pytorch doc)------############################
"""
class Net(nn.Module):

    model_file="models/default_model.pth"
    '''This file will be loaded to test your model. Use --model-file to load/store a different model.'''

    def __init__(self):
        super().__init__()
        # Convolutional layer 1: takes 3-channel input, applies 6 5x5 filters, producing 6 feature maps (3x32x32 -> 6x28x28)
        self.conv1 = nn.Conv2d(3, 6, 5)
        # Max pooling layer: reduces spatial dimensions by half using 2x2 window (6x28x28 -> 6x14x14)
        self.pool = nn.MaxPool2d(2, 2)
        # # Convolutional layer 2: takes 6 feature maps as input, applies 16 5x5 filters, producing 16 feature maps
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Fully connected layer 1: flattens the feature maps from 16x5x5 into a single vector and connects to 120 neurons
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # Fully connected layer 2: connects 120 neurons to 84 neurons
        self.fc2 = nn.Linear(120, 84)
        # Fully connected layer 3 (Output layer): maps 84 neurons to 10 neurons, each representing a class score
        self.fc3 = nn.Linear(84, 10)
            
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # 3x32x32 -> 6x28x28 -> 6x14x14
        x = self.pool(F.relu(self.conv2(x))) # 6x14x14 -> 16x10x10 -> 16x5x5
        x = torch.flatten(x, 1)              # 16x5x5 -> 400
        x = F.relu(self.fc1(x))              # 400 -> 120
        x = F.relu(self.fc2(x))              # 120 -> 84
        x = self.fc3(x)                      # 84 -> 10
        x = F.log_softmax(x, dim=1)          # 10 -> 10, from scores to log-probabilities
        return x

    def save(self, model_file):
        '''Helper function, use it to save the model weights after training.'''
        torch.save(self.state_dict(), model_file)

    def load(self, model_file):
        self.load_state_dict(torch.load(model_file, map_location=torch.device(device)))

        
    def load_for_testing(self, project_dir='./'):
        '''This function will be called automatically before testing your
           project, and will load the model weights from the file
           specify in Net.model_file.
           
           You must not change the prototype of this function. You may
           add extra code in its body if you feel it is necessary, but
           beware that paths of files used in this function should be
           refered relative to the root of your project directory.
        '''        
        self.load(os.path.join(project_dir, Net.model_file))
"""

############################------Improved Randomized Neural Network architecture------############################
class Net(nn.Module):
    
    model_file="models/default_model.pth"
    '''This file will be loaded to test your model. Use --model-file to load/store a different model.'''

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def __init__(self):
        super().__init__()
        # Convolutional layer 1: takes 3-channel input, applies 6 5x5 filters, producing 6 feature maps (3x32x32 -> 6x28x28)
        self.conv1 = nn.Conv2d(3, 6, 5)
        # Max pooling layer: reduces spatial dimensions by half using 2x2 window (6x28x28 -> 6x14x14)
        self.pool = nn.MaxPool2d(2, 2)
        # # Convolutional layer 2: takes 6 feature maps as input, applies 16 5x5 filters, producing 16 feature maps
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Fully connected layer 1: flattens the feature maps from 16x5x5 into a single vector and connects to 120 neurons
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # Fully connected layer 2: connects 120 neurons to 84 neurons
        self.fc2 = nn.Linear(120, 84)
        # Fully connected layer 3 (Output layer): maps 84 neurons to 10 neurons, each representing a class score
        self.fc3 = nn.Linear(84, 10)
        # Dropout layer
        self.dropout = nn.Dropout(p=0.5)
        # Random weight initialization
        self.apply(self.init_weights)
        # Activation functions list
        self.activations = [F.relu, F.leaky_relu]

    def random_activation(self, x):
        '''Apply a random activation function from the predefined list.'''
        act_func = random.choice(self.activations)
        return act_func(x)
    
    def forward(self, x):
        x = self.pool(self.random_activation(self.conv1(x))) # 3x32x32 -> 6x28x28 -> 6x14x14
        x = self.pool(self.random_activation(self.conv2(x))) # 6x14x14 -> 16x10x10 -> 16x5x5
        x = torch.flatten(x, 1)                              # 16x5x5 -> 400
        x = F.relu(self.fc1(x))                              # 400 -> 120
        x = self.dropout(x)                                  # Dropout
        x = F.relu(self.fc2(x))                              # 120 -> 84
        x = self.dropout(x)                                  # Dropout
        x = self.fc3(x)                                      # 84 -> 10
        x = F.log_softmax(x, dim=1)                          # 10 -> 10, from scores to log-probabilities
        return x

    def save(self, model_file):
        '''Helper function, use it to save the model weights after training.'''
        torch.save(self.state_dict(), model_file)

    def load(self, model_file):
        self.load_state_dict(torch.load(model_file, map_location=torch.device(device)))

        
    def load_for_testing(self, project_dir='./'):
        '''This function will be called automatically before testing your
           project, and will load the model weights from the file
           specify in Net.model_file.
           
           You must not change the prototype of this function. You may
           add extra code in its body if you feel it is necessary, but
           beware that paths of files used in this function should be
           refered relative to the root of your project directory.
        '''        
        self.load(os.path.join(project_dir, Net.model_file))


def train_model(net, train_loader, pth_filename, num_epochs, val_natural_loader = None, val_adv_loader = None):
    '''Basic training function (from pytorch doc.)'''
    print("Starting training")
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    val_acc = 0
    val_natural_acc = 0
    val_adv_acc = 0
    best_epoch = 0
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        if epoch %5 == 0: #We will perform validation every 5 epochs
            natural_acc = test(net, val_natural_loader)
            print("Model natural accuracy (valid): {}".format(natural_acc))
            acc = natural_acc

            if val_adv_loader is not None:
                adv_acc = test(net, val_adv_loader)
                print("Model adversial accuracy (valid): {}".format(adv_acc))
                acc = (natural_acc + adv_acc)/2
            
            print("Model average accuracy is ", acc)

            if val_adv_loader is not None and adv_acc > val_adv_acc: #Save model
                val_acc = acc
                val_adv_acc = adv_acc
                val_natural_acc = natural_acc
                best_epoch = epoch
                net.save(pth_filename)
            elif val_adv_loader is None and acc > val_acc:
                val_acc = acc
                val_natural_acc = natural_acc
                best_epoch = epoch
                net.save(pth_filename)

    print("Model best natural accuracy (valid): {}".format(val_natural_acc))
    if val_adv_loader is not None:
        print("Model best adversial accuracy (valid): {}".format(val_adv_acc))
    print("Model best average accuracy is ", val_acc)
    print("Model best epoch is ", best_epoch)


def test(net, test_loader):
    '''Basic testing function.'''

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i,data in enumerate(test_loader, 0):
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

def get_train_loader(dataset, valid_size=1024, batch_size=32):
    '''Split dataset into [train:valid] and return a DataLoader for the training part.'''

    indices = list(range(len(dataset)))
    train_sampler = torch.utils.data.SubsetRandomSampler(indices[valid_size:])
    train = torch.utils.data.DataLoader(dataset, sampler=train_sampler, batch_size=batch_size)

    return train

def get_validation_loader(dataset, valid_size=1024, batch_size=32):
    '''Split dataset into [train:valid] and return a DataLoader for the validation part.'''

    indices = list(range(len(dataset)))
    valid_sampler = torch.utils.data.SubsetRandomSampler(indices[:valid_size])
    valid = torch.utils.data.DataLoader(dataset, sampler=valid_sampler, batch_size=batch_size)

    return valid

def main():

    #### Parse command line arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", default=Net.model_file,
                        help="Name of the file used to load or to sore the model weights."\
                        "If the file exists, the weights will be load from it."\
                        "If the file doesn't exists, or if --force-train is set, training will be performed, "\
                        "and the model weights will be stored in this file."\
                        "Warning: "+Net.model_file+" will be used for testing (see load_for_testing()).")
    parser.add_argument('-f', '--force-train', action="store_true",
                        help="Force training even if model file already exists"\
                             "Warning: previous model file will be erased!).")
    parser.add_argument('-e', '--num-epochs', type=int, default=50,
                        help="Set the number of epochs during training")
    parser.add_argument("-train_adversial", action="store_true",
                        help="if we want to train on the complete dataset (normal + adversial)")
    parser.add_argument("-valid_adversial", action="store_true",
                        help="if we want to valid on the adversial samples")
    args = parser.parse_args()

    #### Create model and move it to whatever device is available (gpu/cpu)
    net = Net()
    net.to(device)

    #### Model training (if necessary)
    if not os.path.exists(args.model_file) or args.force_train:
        print("Training model")
        print(type(args.train_adversial), args.train_adversial)

        if args.train_adversial:
            images = torch.from_numpy(np.load("./data/PGD/images.npy"))
            labels = torch.from_numpy(np.load("./data/PGD/labels.npy"))
            train_dataset = CustomImageDataset(images, labels)
            train_loader = get_train_loader(train_dataset, valid_size=0)
        else:
            # Data Augmentation for Training
            train_transform = transforms.Compose([
                              transforms.RandomCrop(32, padding=4),
                              transforms.RandomHorizontalFlip(),
                              transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                              transforms.ToTensor()])
            cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=train_transform)
            train_loader = get_train_loader(cifar, valid_size, batch_size=batch_size)
        
        #Load datasets for validation
        cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=transforms.ToTensor()) 
        valid_natural_loader = get_validation_loader(cifar, valid_size)
        
        if args.valid_adversial:
            images = torch.from_numpy(np.load("./data/PGD/val_images.npy"))
            labels = torch.from_numpy(np.load("./data/PGD/val_labels.npy"))
            val_dataset = CustomImageDataset(images, labels)
            val_adv_loader = get_validation_loader(val_dataset, valid_size=len(val_dataset))
            train_model(net, train_loader, args.model_file, args.num_epochs, valid_natural_loader, val_adv_loader)
        else:
            train_model(net, train_loader, args.model_file, args.num_epochs, valid_natural_loader, None)

    #### Model testing
    print("Testing with model from '{}'. ".format(args.model_file))

    # Note: You should not change the transform applied to the
    # validation dataset since, it will be the only transform used
    # during final testing.
    cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=transforms.ToTensor()) 
    valid_loader = get_validation_loader(cifar, valid_size)

    net.load(args.model_file)

    acc = test(net, valid_loader)
    print("Model natural accuracy (valid): {}".format(acc))

    if args.valid_adversial:
        #Test accuracy of model on l2 PGD adversial examples
        images = torch.from_numpy(np.load("./data/PGD/val_images.npy"))
        labels = torch.from_numpy(np.load("./data/PGD/val_labels.npy"))
        val_dataset = CustomImageDataset(images, labels)
        val_loader = get_validation_loader(val_dataset, valid_size=len(val_dataset))

        adv_acc_l2 = test(net, val_loader)
        print("Model adversial accuracy for l2 adversial(valid): {}".format(adv_acc_l2))

        #Test accuracy of model on linf PGD adversial examples
        images = torch.from_numpy(np.load("./data/PGD/val_inf_images.npy"))
        labels = torch.from_numpy(np.load("./data/PGD/val_inf_labels.npy"))
        val_dataset = CustomImageDataset(images, labels)
        val_loader = get_validation_loader(val_dataset, valid_size=len(val_dataset))

        adv_acc_inf = test(net, val_loader)
        print("Model adversial accuracy for linf adversial(valid): {}".format(adv_acc_inf))

        acc = ((adv_acc_l2 + adv_acc_inf)/2 + acc)/2

        print("Model average accuracy is ", acc)

    if args.model_file != Net.model_file:
        print("Warning: '{0}' is not the default model file, "\
              "it will not be the one used for testing your project. "\
              "If this is your best model, "\
              "you should rename/link '{0}' to '{1}'.".format(args.model_file, Net.model_file))

if __name__ == "__main__":
    main()

