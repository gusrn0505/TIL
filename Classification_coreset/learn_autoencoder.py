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


from auto_encoder import AutoEncoder, ConvAutoEncoder, ae_train
from active_learn import argparser

if __name__ == "__main__":
    args = argparser().parse_args()
    pprint.pprint(args)
    
    # parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    # default=False 값에서 not을 부여서 True로 만든건가? 
    use_cuda = not args.no_cuda and torch.cuda.is_available()


    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    # use_cuda가 true라면 kwargs를 다음과 같이 지정하기. 
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # 데이터 변경시 수정 필요 
    ae_training_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )
    # 데이터 변경시 수정 필요 
    ae_test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )


    # 데이터 셋의 차원에 따라 수정해야 함 
    AE = AutoEncoder(3* 32 * 32, args.hidden_dim1, args.hidden_dim2)
    AE_loss = nn.MSELoss()

    
    AE = AE.to(device)

    AE_optimizer = optim.Adam(AE.parameters(), lr=args.lr)

    #데이터 변경시 수정 필요. 
    PATH = './weights/CIFAR10/'
    if not os.path.exists(PATH): os.mkdir(PATH)

        # 한번만 Train을 시킬 방법이 없을까? 
    ae_train(AE, ae_training_data, ae_test_data, device, AE_loss, AE_optimizer, args.ae_epochs, kwargs)

    torch.save(AE, PATH + 'AE.pt')  # 전체 모델 저장
    
    torch.save(AE.state_dict(), PATH + 'AE_state_dict.pt')  # 모델 객체의 state_dict 저장
    torch.save({
        'model': AE.state_dict(),
        'optimizer': AE_optimizer.state_dict()
    }, PATH + 'AE_all.tar') 


    """
    CAE = ConvAutoEncoder()
    CAE_loss = nn.MSELoss()

    CAE = CAE.to(device)
    CAE_optimizer = optim.Adam(CAE.parameters(), lr=args.lr)
    ae_train(CAE, ae_training_data, ae_test_data, device, AE_loss, AE_optimizer, args.ae_epochs, kwargs)



    torch.save(CAE, PATH + 'CAE.pt')  # 전체 모델 저장
    
    torch.save(CAE.state_dict(), PATH + 'CAE_state_dict.pt')  # 모델 객체의 state_dict 저장
    torch.save({
        'model': CAE.state_dict(),
        'optimizer': CAE_optimizer.state_dict()
    }, PATH + 'CAE_all.tar') 
    """


    print("Finish training AE, CAE")