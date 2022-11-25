from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

class AutoEncoder(nn.Module):
  def __init__(self, input_dim, hidden_dim1, hidden_dim2):
    super(AutoEncoder, self).__init__()
    
    self.encoder = nn.Sequential(
      nn.Linear(input_dim, hidden_dim1),
      nn.ReLU(),
      nn.Linear(hidden_dim1, hidden_dim2),
      nn.ReLU()
    )
    
    self.decoder = nn.Sequential(
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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # (출력 채널 수, 입력 채널 수, 높이, 너비)를 입력으로 넣어, 
        # (데이터 개수, 출력 채널 수, repre의 높이, repre의 폭) 의 형태로 결과 반환
        # 또는 (batch size, input 채널 수, 높이, 폭) => (배치 사이즈, output 채널 수, output 높이, output 폭) 으로 반환
        
        # 1개의 입력 채널(이미지)를 받아들이고, 사각 커널 사이즈가 3인 32개의 합성곱 특징들을 출력한다. 
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # 32개의 입력 계층을 받아들이고, 사각 커널 사이즈가 3인 합성곱 특징을 출력합니다. 
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128) # 9218 차원에서 128 차원으로 줄이기
        self.fc2 = nn.Linear(128, 10)

    # 기본적인 NN 모델에서 dim 128개로 만든 다음 학습. 
    def get_features(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x) 
        return x
    
    def stochastic_pred(self, x):
        # add dropouts everywhere
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
