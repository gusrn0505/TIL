import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.models as models

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import numpy as np
from random import sample
import matplotlib.pyplot as plt




_ARCH_REGISTRY = {}


def architecture(name, sample_shape):
    """
    Decorator to register an architecture;

    Use like so:

    >>> @architecture('my_architecture', (3, 32, 32))
    ... class MyNetwork(nn.Module):
    ...     def __init__(self, n_classes):
    ...         # Build network
    ...         pass
    """
    def decorate(fn):
        _ARCH_REGISTRY[name] = (fn, sample_shape)
        return fn
    return decorate


def get_net_and_shape_for_architecture(arch_name):
    """
    Get network building function and expected sample shape:

    For example:
    >>> net_class, shape = get_net_and_shape_for_architecture('my_architecture')

    >>> if shape != expected_shape:
    ...     raise Exception('Incorrect shape')
    """
    return _ARCH_REGISTRY[arch_name]




@architecture('mnist-bn-32-64-256', (1, 28, 28))
class MNIST_BN_32_64_256 (nn.Module):
    def __init__(self, n_classes, dim_reduction):
        super(MNIST_BN_32_64_256, self).__init__()

        self.conv1_1 = nn.Conv2d(1, 32, (5, 5))
        self.conv1_1_bn = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2_1 = nn.Conv2d(32, 64, (3, 3))
        self.conv2_1_bn = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, (3, 3))
        self.conv2_2_bn = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2, 2))

        self.drop1 = nn.Dropout()
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, n_classes)
        self.fc5 = nn.Linear(n_classes, dim_reduction)


        self.fc_d1 = nn.Linear(dim_reduction, n_classes)
        self.fc_d2 = nn.Linear(n_classes, 256)
        self.fc_d3 = nn.Linear(256, 1024)

        self.convT1= nn.ConvTranspose2d(64, 64, (4,4), 2) # (64, 4, 4) -> (64, 10, 10)
        self.convT1_bn = nn.BatchNorm2d(64)        
        self.convT2  = nn.ConvTranspose2d(64, 32, (3,3)) # (64, 10, 10) -> (32, 12, 12)
        self.convT2_bn = nn.BatchNorm2d(32)
        self.convT3 = nn.ConvTranspose2d(32, 1, (6,6), 2)
        

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1_1_bn(self.conv1_1(x)))) #(1,28,28) -> #(32, 12,12)
        x = F.relu(self.conv2_1_bn(self.conv2_1(x))) # (32,12,12) -> (64, 10, 10)
        x = self.pool2(F.relu(self.conv2_2_bn(self.conv2_2(x)))) #(64, 10, 10) -> (64,4,4)
        x = x.view(-1, 1024) # Flatten 
        x = self.drop1(x)
        x = F.relu(self.fc3(x)) #(1024 -> 256) 
        x = F.relu(self.fc4(x)) # 256 -> n_classes
        x = self.fc5(x) 
        
        
        x = self.fc_d1(x)
        x = F.relu(self.fc_d2(x))
        x = self.drop1(x)
        x = F.relu(self.fc_d3(x)) # 256 -> 1024 
        x = x.view(-1, 64, 4, 4) # 1024 -> (64, 4, 4)
        x = F.relu(self.convT1_bn(self.convT1(x)))
        x = F.relu(self.convT2_bn(self.convT2(x)))
        x = F.relu(self.convT3(x))
        return x




    def get_codes(self, x):
        x = self.pool1(F.relu(self.conv1_1_bn(self.conv1_1(x)))) #(1,28,28) -> #(32, 12,12)
        x = F.relu(self.conv2_1_bn(self.conv2_1(x))) # (32,12,12) -> (64, 10, 10)
        x = self.pool2(F.relu(self.conv2_2_bn(self.conv2_2(x)))) #(64, 10, 10) -> (64,4,4)
        x = x.view(-1, 1024) # Flatten 
        x = self.drop1(x)
        x = F.relu(self.fc3(x)) #(1024 -> 256) 
        x = F.relu(self.fc4(x)) # 256 -> n_classes
        x = self.fc5(x) 
        return x


@architecture('rgb-48-96-192-gp', (3, 32, 32))
class RGB_48_96_192_gp (nn.Module):
    def __init__(self, n_classes, dim_reduction):
        super(RGB_48_96_192_gp, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 48, (3, 3), padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(48)
        self.conv1_2 = nn.Conv2d(48, 48, (3, 3), padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(48)
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2_1 = nn.Conv2d(48, 96, (3, 3), padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(96)
        self.conv2_2 = nn.Conv2d(96, 96, (3, 3), padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(96)
        self.conv2_3 = nn.Conv2d(96, 96, (3, 3), padding=1)
        self.conv2_3_bn = nn.BatchNorm2d(96)
        self.pool2 = nn.MaxPool2d((2, 2))

        self.conv3_1 = nn.Conv2d(96, 192, (3, 3), padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(192)
        self.conv3_2 = nn.Conv2d(192, 192, (3, 3), padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(192)
        self.conv3_3 = nn.Conv2d(192, 192, (3, 3), padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(192)
        self.pool3 = nn.MaxPool2d((2, 2))

        self.drop1 = nn.Dropout()

        self.fc1 = nn.Linear(192, 192)
        self.fc2 = nn.Linear(192, n_classes)
        self.fc3 = nn.Linear(n_classes, dim_reduction)


        self.fc_d1 = nn.Linear(dim_reduction, n_classes)
        self.fc_d2 = nn.Linear(n_classes, 192)
        self.fc_d3 = nn.Linear(192, 192)



        self.convT1= nn.ConvTranspose2d(192, 192, (4,4)) # (192, 1, 1) -> (192, 4, 4)
        self.convT1_bn = nn.BatchNorm2d(192)        
        self.convT2  = nn.ConvTranspose2d(192, 96, (2,2), 2) # (192, 4, 4) -> (96, 8, 8)
        self.convT2_bn = nn.BatchNorm2d(96)
        self.convT3 = nn.ConvTranspose2d(96, 48, (2,2), 2) # (96,8,8) => (48, 16,16)
        self.convT3_bn = nn.BatchNorm2d(48) 
        self.convT4 = nn.ConvTranspose2d(48, 3, (2,2), 2) # (48,16,16) => (3, 32,32)


    def forward(self, x):
        x = F.relu(self.conv1_1_bn(self.conv1_1(x))) # (3,32,32) -> (48,32,32)
        x = self.pool1(F.relu(self.conv1_2_bn(self.conv1_2(x)))) # (48,32,32) => (48,16,16)

        x = F.relu(self.conv2_1_bn(self.conv2_1(x))) #(48,16,16) => (96,16,16)
        x = F.relu(self.conv2_2_bn(self.conv2_2(x))) # (96,16,16) => (96,16,16)
        x = self.pool2(F.relu(self.conv2_3_bn(self.conv2_3(x)))) # (96,16,16) => (96,8,8)

        x = F.relu(self.conv3_1_bn(self.conv3_1(x))) # (96,8,8) => (192, 8,8)
        x = F.relu(self.conv3_2_bn(self.conv3_2(x))) # (192, 8,8) => (192, 8,8)
        x = self.pool3(F.relu(self.conv3_3_bn(self.conv3_3(x)))) # (192, 8, 8) => (192, 4,4)

        x = F.avg_pool2d(x, 4) # (192, 4, 4) => (192, 1, 1) # 이거 오토 encoder로 괜찮나? 일단 해보고 오류면 제외해보자. 
        x = x.view(-1, 192)

        x = self.drop1(x)
        x = F.relu(self.fc1(x)) # (192, 192)
        x = F.relu(self.fc2(x)) # 192 -> n_classes
        x = self.fc3(x)


        x = self.fc_d1(x)
        x = F.relu(self.fc_d2(x))
        x = self.drop1(x)
        x = F.relu(self.fc_d3(x)) # 

        x = x.view(-1, 192, 1, 1) # 192 -> (192, 1, 1)
        x = F.relu(self.convT1_bn(self.convT1(x)))
        x = F.relu(self.convT2_bn(self.convT2(x)))
        x = F.relu(self.convT3_bn(self.convT3(x))) 
        x = self.convT4(x)
        return x

    def get_codes(self, x):
        x = F.relu(self.conv1_1_bn(self.conv1_1(x))) # (3,32,32) -> (48,32,32)
        x = self.pool1(F.relu(self.conv1_2_bn(self.conv1_2(x)))) # (48,32,32) => (48,16,16)

        x = F.relu(self.conv2_1_bn(self.conv2_1(x))) #(48,16,16) => (96,16,16)
        x = F.relu(self.conv2_2_bn(self.conv2_2(x))) # (96,16,16) => (96,16,16)
        x = self.pool2(F.relu(self.conv2_3_bn(self.conv2_3(x)))) # (96,16,16) => (96,8,8)

        x = F.relu(self.conv3_1_bn(self.conv3_1(x))) # (96,8,8) => (192, 8,8)
        x = F.relu(self.conv3_2_bn(self.conv3_2(x))) # (192, 8,8) => (192, 8,8)
        x = self.pool3(F.relu(self.conv3_3_bn(self.conv3_3(x)))) # (192, 8, 8) => (192, 4,4)

        x = F.avg_pool2d(x, 4) # (192, 4, 4) => (192, 1, 1) # 이거 오토 encoder로 괜찮나? 일단 해보고 오류면 제외해보자. 
        x = x.view(-1, 192)

        x = self.drop1(x)
        x = F.relu(self.fc1(x)) # (192, 192)
        x = F.relu(self.fc2(x)) # 192 -> n_classes
        x = self.fc3(x) # n_classes -> dim_reduction 
        return x


@architecture('rgb-128-256-down-gp', (3, 32, 32))
class RGB_128_256_down_gp (nn.Module):
    def __init__(self, n_classes, dim_reduction):
        super(RGB_128_256_down_gp, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 128, (3, 3), padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(128)
        self.conv1_2 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(128)
        self.conv1_3 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv1_3_bn = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.drop1 = nn.Dropout()

        self.conv2_1 = nn.Conv2d(128, 256, (3, 3), padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(256)
        self.conv2_2 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(256)
        self.conv2_3 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.conv2_3_bn = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d((2, 2))
        self.drop2 = nn.Dropout()

        self.conv3_1 = nn.Conv2d(256, 512, (3, 3), padding=0)
        self.conv3_1_bn = nn.BatchNorm2d(512)
        self.conv3_2 = nn.Conv2d(512, 256, (1, 1), padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 128, (1, 1), padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128, n_classes)
        self.fc2 = nn.Linear(n_classes, dim_reduction)


        self.fc_d1 = nn.Linear(dim_reduction, n_classes)
        self.fc_d2 = nn.Linear(n_classes, 128)



        self.convT1= nn.ConvTranspose2d(128, 128, (6,6)) # (128, 1, 1) -> (128, 6, 6)
        self.convT1_bn = nn.BatchNorm2d(128)        
        self.convT2  = nn.ConvTranspose2d(128, 256, (2,2), 2, 2) # (128, 6, 6) -> (256, 8, 8)
        self.convT2_bn = nn.BatchNorm2d(256)
        self.convT3 = nn.ConvTranspose2d(256, 128, (2,2), 2) # (256,8,8) => (128, 16,16)
        self.convT3_bn = nn.BatchNorm2d(128) 
        self.convT4 = nn.ConvTranspose2d(128, 3, (2,2), 2) # (128,16,16) => (3, 32,32)


    def forward(self, x):
        x = F.relu(self.conv1_1_bn(self.conv1_1(x))) 
        x = F.relu(self.conv1_2_bn(self.conv1_2(x))) 
        x = self.pool1(F.relu(self.conv1_3_bn(self.conv1_3(x)))) # (3,32,32) -> (128,16,16)
        x = self.drop1(x)

        x = F.relu(self.conv2_1_bn(self.conv2_1(x))) 
        x = F.relu(self.conv2_2_bn(self.conv2_2(x))) 
        x = self.pool2(F.relu(self.conv2_3_bn(self.conv2_3(x)))) # => (256,8,8)
        x = self.drop2(x)

        x = F.relu(self.conv3_1_bn(self.conv3_1(x))) # (256,8,8) => (512, 6,6)
        x = F.relu(self.conv3_2_bn(self.conv3_2(x))) # (512, 6,6) => (256, 8, 8) 
        x = F.relu(self.conv3_3_bn(self.conv3_3(x))) # (256, 8,8) => (128, 10, 10)

        x = F.avg_pool2d(x, 6) # (128, 1, 1) ?   
        x = x.view(-1, 128)

        x = F.relu(self.fc1(x)) # (128 => 10)
        x = self.fc2(x) # 10 -> 3
    

        x = self.fc_d1(x)
        x = F.relu(self.fc_d2(x))
        x = self.drop1(x) 

        x = x.view(-1, 128, 1, 1) # 128 -> (128, 1, 1)
        x = F.relu(self.convT1_bn(self.convT1(x)))
        x = F.relu(self.convT2_bn(self.convT2(x)))
        x = F.relu(self.convT3_bn(self.convT3(x))) 
        x = self.convT4(x)
        return x

    def get_codes(self, x):
        x = F.relu(self.conv1_1_bn(self.conv1_1(x))) 
        x = F.relu(self.conv1_2_bn(self.conv1_2(x))) 
        x = self.pool1(F.relu(self.conv1_3_bn(self.conv1_3(x)))) # (3,32,32) -> (128,16,16)
        x = self.drop1(x)

        x = F.relu(self.conv2_1_bn(self.conv2_1(x))) 
        x = F.relu(self.conv2_2_bn(self.conv2_2(x))) 
        x = self.pool2(F.relu(self.conv2_3_bn(self.conv2_3(x)))) # => (256,8,8)
        x = self.drop2(x)

        x = F.relu(self.conv3_1_bn(self.conv3_1(x))) # (256,8,8) => (512, 6,6)
        x = F.relu(self.conv3_2_bn(self.conv3_2(x))) # (512, 6,6) => (256, 8, 8) 
        x = F.relu(self.conv3_3_bn(self.conv3_3(x))) # (256, 8,8) => (128, 10, 10)

        x = F.avg_pool2d(x, 6) # (128, 1, 1) ?   
        x = x.view(-1, 128)

        x = F.relu(self.fc1(x)) # (128 => 10)
        x = self.fc2(x) # 10 -> 3
        return x




def ae_train(model, training_data, test_data, device, Loss, optimizer, num_epochs, kwargs):
    train_loss_arr = []
    test_loss_arr = []

    best_test_loss = 99999999
    early_stop, early_stop_max = 0., 3.

    
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=False, **kwargs)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False, **kwargs)

    for epoch in range(num_epochs):

        epoch_loss = 0.
        for batch_X, _ in train_dataloader:
      
            batch_X = batch_X.to(device)
            optimizer.zero_grad()

      # Forward Pass
            model.train()
            outputs = model(batch_X)
            train_loss = Loss(outputs, batch_X)
            epoch_loss += train_loss.data

      # Backward and optimize
            train_loss.backward()
            optimizer.step()

        train_loss_arr.append(epoch_loss / train_dataloader.batch_size)

        if epoch % 5 == 0:
            model.eval()

            test_loss = 0.

            for batch_X, _ in test_dataloader: 
                batch_X = batch_X.to(device)

        # Forward Pass
                outputs = model(batch_X)
                batch_loss = Loss(outputs, batch_X)
                test_loss += batch_loss.data

            test_loss = test_loss
            test_loss_arr.append(test_loss)

            if best_test_loss > test_loss:
                best_test_loss = test_loss
                early_stop = 0

                print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f} *'.format(epoch, num_epochs, epoch_loss, test_loss))
        
            else: 
                early_stop += 1
                print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch, num_epochs, epoch_loss, test_loss))   

        if early_stop >= early_stop_max: break



## 모델 미리 학습시키기##

