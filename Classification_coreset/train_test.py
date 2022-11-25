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
    for i, data in enumerate(labeled_dataset):
        target = labeled_dataset_label[i]

        data, target = data.to(device), target.to(device)
        # pdb.set_trace()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i, len(labeled_dataset),
                100. * i / len(labeled_dataset), loss.item()))
    return model

"""
def test(args, model, device, test_dataset, test_dataset_label, optimizer1, epoch) : 
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for sample in test_loader:
            data = sample['image']
            target = sample['label']

            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return 100. * correct/len(test_loader.dataset)
"""
