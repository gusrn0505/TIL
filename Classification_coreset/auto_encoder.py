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




class AutoEncoder(nn.Module):
  def __init__(self, input_dim, hidden_dim1, hidden_dim2):
    super(AutoEncoder, self).__init__()
    
    self.encoder = nn.Sequential(
      nn.Linear(input_dim, hidden_dim1),
      nn.ReLU(),
      nn.Linear(hidden_dim1, hidden_dim2),
      nn.ReLU(),
      nn.Linear(hidden_dim2, 8),
      nn.ReLU(),
      nn.Linear(8, 2),
      nn.ReLU()
    )
    
    self.decoder = nn.Sequential(
      nn.Linear(2, 8),
      nn.ReLU(),
      nn.Linear(8, hidden_dim2),
      nn.ReLU(),
      nn.Linear(hidden_dim2, hidden_dim1),
      nn.ReLU(),
      nn.Linear(hidden_dim1, input_dim),
      nn.ReLU()
    )
  
  def forward(self, x):
    out = x.view(x.size(0), -1)
    out = self.encoder(out)
    out = self.decoder(out)
    out = out.view(x.size())
    return out
  
  def get_codes(self, x):
    return self.encoder(x)



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








class ConvAutoEncoder(nn.Module):
  def __init__(self, input_size, cnn_kernel, cnn_stride, cnn_padding):
    super(ConvAutoEncoder, self).__init__()
    
    # Encoder
    self.cnn_layer1 = nn.Sequential(
      
      nn.Conv2d(1, 16, kernel_size = cnn_kernel, stride=cnn_stride, padding=2),
      #nn.Conv2d(3, 16, kernel_size = cnn_kernel, stride=cnn_stride, padding=cnn_padding),
      nn.ReLU(),
      nn.BatchNorm2d(16), 
      nn.Conv2d(16, 32, kernel_size = cnn_kernel, stride=cnn_stride, padding=cnn_padding),
      nn.ReLU(), 
      nn.BatchNorm2d(32), 
      nn.Conv2d(32, 64, kernel_size = cnn_kernel, stride=cnn_stride, padding=cnn_padding),
      nn.ReLU(), 
      nn.BatchNorm2d(64),
      nn.MaxPool2d(2,2)
      )


    self.fc_encoder = nn.Sequential(
      nn.Linear(256, 64),
      nn.ReLU(),
      nn.Linear(64, 8),
      nn.ReLU(),
      nn.Linear(8, 2),
      nn.ReLU()
    )

    self.fc_decoder = nn.Sequential(
      nn.Linear(2, 8),
      nn.ReLU(),
      nn.Linear(8, 64),
      nn.ReLU(),
      nn.Linear(64, 256),
      nn.ReLU()
    )

    # Decoder 
    # ConvTranspose2d : output H/W = Kernel size + stride(input size -1) - 2 padding
    # (N, 3, cnn2_size, cnn2_size) => (N, 3, cnn1_size, cnn1_size)
    self.tran_cnn_layer1 = nn.Sequential(
      nn.ConvTranspose2d(64, 64, kernel_size = cnn_kernel, stride =cnn_stride, padding =cnn_padding),
      nn.ReLU(),
      nn.BatchNorm2d(64), 
      nn.ConvTranspose2d(64, 32, kernel_size = cnn_kernel, stride =cnn_stride, padding =cnn_padding),
      nn.ReLU(),
      nn.BatchNorm2d(32),
      nn.ConvTranspose2d(32, 16, kernel_size = cnn_kernel, stride =cnn_stride, padding =cnn_padding),
      nn.ReLU(),
      nn.BatchNorm2d(16),
      nn.ConvTranspose2d(16, 1, kernel_size = cnn_kernel, stride =cnn_stride, padding =2),
      #nn.ConvTranspose2d(16, 3, kernel_size = cnn_kernel, stride =cnn_stride, padding =cnn_padding),
      nn.ReLU()
    )

    
  def forward(self, x):
    out = self.cnn_layer1(x)
    out = torch.flatten(out, 1) # batchsize - 64, 32 * cnn2^2  
    out = self.fc_encoder(out) # 64, 32 * cnn2^2 -> 64, 2
    out = self.fc_decoder(out) # 64, 2 => 64, 32 * cnn2^2 2048
    out = out.view(len(x), 64, 2, 2)  # (batch_size, , H, W)
    out = self.tran_cnn_layer1(out)
    return out

  def get_codes(self, x):
    out = self.cnn_layer1(x) 
    out = torch.flatten(out, 1)
    out = self.fc_encoder(out)
    return out 


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

        if epoch % 10 == 0:
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

        if early_stop >= early_stop_max: 
            break



## 모델 미리 학습시키기##

