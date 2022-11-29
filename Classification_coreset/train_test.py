from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.models as models
import pdb

from dsets.mnist import MNIST
from mymodels.mnist_net import Net

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


def train(args, model, device, labeled_dataset, labeled_dataset_label, optimizer, epoch):
    model.train()
    
    # for 절 자체에 오류가 나는 걸까, 아님 아래의 내용들에서 문제가 생기는 걸까. 
    # batch learning을 해주고 싶은데 흠. 
    labeled_dataset = torch.tensor(labeled_dataset)
    labeled_dataset_label = torch.tensor(labeled_dataset_label)
    
    all_data = [(labeled_dataset[i], labeled_dataset_label[i]) for i in range(len(labeled_dataset_label))]
    data_loader = DataLoader(all_data, batch_size=32)

    for i, (data, target) in enumerate(data_loader):
        data = data.view(len(target), 1, 28,28)
        data, target = data.to(device), target.to(device)
        # pdb.set_trace()
        optimizer.zero_grad()
        output = model(data) #여기가 문제가 생기는 지점 
        loss = F.nll_loss(output, target) 
        loss.backward()
        optimizer.step()
        if i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i, len(data_loader),
                100. * i / len(data_loader), loss.item()))
    return model

def test(args, model, device, test_dataset, test_dataset_label, optimizer1, epoch) :
    model.eval()
    test_loss = 0
    correct = 0


    test_dataset = torch.tensor(test_dataset)
    test_dataset_label = torch.tensor(test_dataset_label)

    all_data = [(test_dataset[i], test_dataset_label[i]) for i in range(len(test_dataset_label))]
    data_loader = DataLoader(all_data, batch_size=args.batch_size)

    # dataloader에 index가 가능한가? 
    with torch.no_grad():
        for data, target in data_loader:
            data = data.view(len(target), 1, 28,28)
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_dataset),
        100. * correct / len(test_dataset)))

    return correct/len(test_dataset)

