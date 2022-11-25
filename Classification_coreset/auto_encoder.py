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
  def __init__(self):
    super(ConvAutoEncoder, self).__init__()
    
    self.encoder = nn.Sequential(
      nn.Conv2d(1, 3, kernel_size = 5),
      nn.ReLU(),
      nn.Conv2d(3, 5, kernel_size = 5),
      nn.ReLU()
    )
    
    self.decoder = nn.Sequential(
      nn.ConvTranspose2d(5, 3, kernel_size = 5),
      nn.ReLU(),
      nn.ConvTranspose2d(3, 1, kernel_size = 5),
      nn.ReLU()
    )
    
  def forward(self, x):
    out = self.encoder(x)
    out = self.decoder(out)
    return out

  def get_codes(self, x):
    return self.encoder(x)


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

        train_loss_arr.append(epoch_loss / len(train_dataloader.dataset))

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

