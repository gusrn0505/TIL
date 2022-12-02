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

