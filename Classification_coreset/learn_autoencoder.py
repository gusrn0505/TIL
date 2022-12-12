from __future__ import print_function, division
import os
import torch


# DataLoader은 Dataset을 샘플에 쉽게 접근할 수 있도록 순회가능한 객체(iterable)로 감쌉니다
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
from torchvision.transforms import ToTensor
import torchvision.models as models 

import pprint
from datetime import datetime



import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from auto_encoder import MNIST_BN_32_64_256, RGB_48_96_192_gp  , RGB_128_256_down_gp, ae_train

if __name__ == "__main__":
    use_cuda = True
    torch.manual_seed(20)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # 데이터 변경시 수정 필요 
    ae_training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )
    # 데이터 변경시 수정 필요 
    ae_test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )


    # 데이터 셋의 차원에 따라 수정해야 함 
    AE = MNIST_BN_32_64_256(10, 2)
    #AE = RGB_48_96_192_gp(10,3)
    #AE = RGB_128_256_down_gp(10,3)
    AE_loss = nn.MSELoss()
    AE = AE.to(device)

    #AE_optimizer = optim.Adam(AE.parameters(), lr=args.lr)
    AE_optimizer = optim.Adam(AE.parameters(), lr=0.001)
    

    #데이터 변경시 수정 필요. 
    PATH = './weights/FashionMNIST/'
    if not os.path.exists(PATH): os.mkdir(PATH)

        # 한번만 Train을 시킬 방법이 없을까? 
    #ae_train(AE, ae_training_data, ae_test_data, device, AE_loss, AE_optimizer, args.ae_epochs, kwargs)
    ae_train(AE, ae_training_data, ae_test_data, device, AE_loss, AE_optimizer, 100, kwargs)
    torch.save(AE, PATH + 'CAE.pt')  # 전체 모델 저장
    
    torch.save(AE.state_dict(), PATH + 'CAE_state_dict.pt')  # 모델 객체의 state_dict 저장
    torch.save({
        'model': AE.state_dict(),
        'optimizer': AE_optimizer.state_dict()
    }, PATH + 'CAE_all.tar') 


    print("Finish training AE, CAE")