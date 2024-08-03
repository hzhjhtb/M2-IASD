
#
#
#      0===========================================================0
#      |       TP6 PointNet for point cloud classification         |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Jean-Emmanuel DESCHAUD - 21/02/2023
#

import numpy as np
import random
import math
import os
import time
import torch
import scipy.spatial.distance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import sys

# Import functions to read and write ply files
from ply import write_ply, read_ply



class RandomRotation_z(object):
    def __call__(self, pointcloud):
        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[ math.cos(theta), -math.sin(theta),      0],
                               [ math.sin(theta),  math.cos(theta),      0],
                               [0,                               0,      1]])
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return rot_pointcloud

class RandomNoise(object):
    def __call__(self, pointcloud):
        noise = np.random.normal(0, 0.02, (pointcloud.shape))
        noisy_pointcloud = pointcloud + noise
        return noisy_pointcloud
    
class RandomScaling(object):
    def __call__(self, pointcloud):
        scale = np.random.uniform(low=0.8, high=1.2)  # Random scaling factor
        scaled_pointcloud = pointcloud * scale
        return scaled_pointcloud
        

        
class ToTensor(object):
    def __call__(self, pointcloud):
        return torch.from_numpy(pointcloud)


def default_transforms():
    #return transforms.Compose([RandomRotation_z(),RandomNoise(),ToTensor()])
    return transforms.Compose([
            RandomRotation_z(),
            RandomNoise(),
            RandomScaling(),  # Include the new augmentation
            ToTensor()
        ])

def test_transforms():
    return transforms.Compose([ToTensor()])



class PointCloudData_RAM(Dataset):
    def __init__(self, root_dir, folder="train", transform=default_transforms()):
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir+"/"+dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform
        self.data = []
        for category in self.classes.keys():
            new_dir = root_dir+"/"+category+"/"+folder
            for file in os.listdir(new_dir):
                if file.endswith('.ply'):
                    ply_path = new_dir+"/"+file
                    data = read_ply(ply_path)
                    sample = {}
                    sample['pointcloud'] = np.vstack((data['x'], data['y'], data['z'])).T
                    sample['category'] = self.classes[category]
                    self.data.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pointcloud = self.transforms(self.data[idx]['pointcloud'])
        return {'pointcloud': pointcloud, 'category': self.data[idx]['category']}



class MLP(nn.Module):
    def __init__(self, classes = 10):
        super(MLP, self).__init__()

        # The flatten layer
        self.flatten = nn.Flatten()

        # The first layer
        self.fc1 = nn.Linear(3072, 512)
        self.bn1 = nn.BatchNorm1d(512)

        # The second layer
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p = 0.3)

        # The output layer
        self.fc3 = nn.Linear(256, classes)
        

    def forward(self, input):
        # Flatten the input tensor
        x = self.flatten(input)

        # The first layer
        x = F.relu(self.bn1(self.fc1(x)))

        # The second layer
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        # The output layer
        x = self.fc3(x)

        return x




class PointNetBasic(nn.Module):
    def __init__(self, classes=10):
        super(PointNetBasic, self).__init__()
        # Shared MLP
        self.conv1 = nn.Conv1d(3, 64, 1)  # Assumes input point cloud of size [batch_size, num_points, 3]
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        
        # MLP for global features
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        
        # Max pooling
        self.mp = nn.MaxPool1d(1024)
        
        # Final MLP
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        
        # Flatten layer
        self.flatten = nn.Flatten()

    def forward(self, input):
        # Shared MLP
        x = F.relu(self.bn1(self.conv1(input)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # MLP for global features
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        
        # Max pooling
        x = self.mp(x)
        
        # Flatten the features for the fully connected layers
        x = self.flatten(x)
        
        # Final MLP
        x = F.relu(self.bn6(self.fc1(x)))
        x = F.relu(self.bn7(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        
        return x

        
        
        
class Tnet(nn.Module):
    def __init__(self, k=3):
        super(Tnet, self).__init__()

        # Shared MLP
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        # Max pooling
        self.mp = nn.MaxPool1d(1024)

        # Final MLP
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        # Flatten layer
        self.flatten = nn.Flatten()


    def forward(self, input):
        # Shared MLP
        x = F.relu(self.bn1(self.conv1(input)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Max pooling
        x = self.mp(x)
        
        # Flatten the features for the fully connected layers
        x = self.flatten(x)

        # Final MLP
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # # Add an identity matrix so that the network outputs a transformation close to an identity matrix
        iden = torch.eye(self.k, device=x.device)
        iden = iden.repeat(x.size(0), 1, 1)
        x = x.view(-1, self.k, self.k) + iden

        return x


class PointNetFull(nn.Module):
    def __init__(self, classes = 10):
        super(PointNetFull, self).__init__()
        
        # First Tnet output a 3x3 matrix to algin 3d point clouds
        self.tnet1 = Tnet(k=3)

        # Shared MLP
        self.conv1 = nn.Conv1d(3, 64, 1)  # Assumes input point cloud of size [batch_size, num_points, 3]
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        #A new Tnet output a 64x64 matrix to algin points in feature space of dim 64
        self.tnet2 = Tnet(k=64)
        
        # MLP for global features
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        
        # Max pooling
        self.mp = nn.MaxPool1d(1024)
        
        # Final MLP
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        
        # Flatten layer
        self.flatten = nn.Flatten()


    def forward(self, input):
        # First Tnet
        trans1 = self.tnet1(input)
        x = torch.bmm(input.transpose(1,2), trans1).transpose(1,2)

        # Shared MLP
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Second Tnet
        trans2 = self.tnet2(x)
        x = torch.bmm(x.transpose(1,2), trans2).transpose(1,2)

        # MLP for global features
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        
        # Max pooling
        x = self.mp(x)
        
        # Flatten the features for the fully connected layers
        x = self.flatten(x)
        
        # Final MLP
        x = F.relu(self.bn6(self.fc1(x)))
        x = F.relu(self.bn7(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        
        return x, trans1



def basic_loss(outputs, labels):
    criterion = torch.nn.CrossEntropyLoss()
    return criterion(outputs, labels)

def pointnet_full_loss(outputs, labels, m3x3, alpha = 0.001):
    criterion = torch.nn.CrossEntropyLoss()
    bs=outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs,1,1)
    if outputs.is_cuda:
        id3x3=id3x3.cuda()
    diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
    return criterion(outputs, labels) + alpha * (torch.norm(diff3x3)) / float(bs)



def train(model, device, train_loader, test_loader=None, epochs=250):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    loss=0
    for epoch in range(epochs): 
        model.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
            optimizer.zero_grad()
            #outputs = model(inputs.transpose(1,2))
            outputs, m3x3 = model(inputs.transpose(1,2))
            #loss = basic_loss(outputs, labels)
            loss = pointnet_full_loss(outputs, labels, m3x3)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct = total = 0
        test_acc = 0
        if test_loader:
            with torch.no_grad():
                for data in test_loader:
                    inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                    #outputs = model(inputs.transpose(1,2))
                    outputs, __ = model(inputs.transpose(1,2))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            test_acc = 100. * correct / total
            print('Epoch: %d, Loss: %.3f, Test accuracy: %.1f %%' %(epoch+1, loss, test_acc))


 
if __name__ == '__main__':
    
    t0 = time.time()
    
    ROOT_DIR = "../data/ModelNet10_PLY"
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    print("Device: ", device)
    
    train_ds = PointCloudData_RAM(ROOT_DIR, folder='train', transform=default_transforms())
    test_ds = PointCloudData_RAM(ROOT_DIR, folder='test', transform=test_transforms())

    inv_classes = {i: cat for cat, i in train_ds.classes.items()}
    print("Classes: ", inv_classes)
    print('Train dataset size: ', len(train_ds))
    print('Test dataset size: ', len(test_ds))
    print('Number of classes: ', len(train_ds.classes))
    print('Sample pointcloud shape: ', train_ds[0]['pointcloud'].size())

    train_loader = DataLoader(dataset=train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_ds, batch_size=32)

    #model = MLP()
    #model = PointNetBasic()
    model = PointNetFull()
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print("Number of parameters in the Neural Networks: ", sum([np.prod(p.size()) for p in model_parameters]))
    model.to(device)
    
    train(model, device, train_loader, test_loader, epochs = 10)
    
    t1 = time.time()
    print("Total time for training : ", t1-t0)

    
    


