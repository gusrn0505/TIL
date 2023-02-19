import numpy as np
# MKL_SERVICE_FORCE_INTEL=1
import sys

import os
import os.path as osp
import torch
import torchvision
import time

import pandas as pd
import copy
import argparse
import random
import shutil
import warnings
import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.utils.data
from tqdm.notebook import tqdm
import chicken_dataset as dataset_

def parameter_tuning(mu, sigma) : 
    total_val_auc = 0
    epoch = 3
    from_=time.time()
    for epoch in range(epoch):
        
        ## train 학습 수행
        epoch_loss, disc_loss, gen_loss=U.Train(train_loader, models, criterions, optimizers,mu,sigma,device=device, log=log, args=args)
        if (args.optim=='SGD'):
            encoder_scheduler.step(epoch_loss)
            decoder_scheduler.step(epoch_loss)
            discriminator_scheduler.step(disc_loss)
            generator_schedular.step(gen_loss)
            
        ## validation 수행
        ## 여기서 val_loss가 최적의 대상이 되어야하지 않을까? -> 모형이 학습이 되는 기준이므로
        val_loss, val_recall, val_best_recall_threshold, val_auc, val_auprc=U.Validation(val_loader, models, criterions, device=device, log=log, args=args)
        total_val_auc += val_auc
    
    ## 도출된 평균 val_loss를 기준으로?
    average_val_auc = total_val_auc/epoch
    to_=time.time()
    log(f'\Epoch Running Time: {int((to_-from_)//60)}m {int((to_-from_)%60)}s')
    
    return average_val_auc


#### Parser
parser=argparse.ArgumentParser(description='training pipeline')
parser.add_argument('--log_path', default='./log', type=str)  # 로그 텍스트를 저장할 위치
parser.add_argument('--gpu', type=str, default=0, help='gpu allocation')  # 사용할 GPU 선택
parser.add_argument('--invert', default=True, help='using inverted mse loss for anomaly')  # Loss inversion 사용 여부
parser.add_argument('--FGSM', help='whether to use FGSM attack')  # FGSM Attack 사용 여부 -> Attack 된 이미지 = Abnormal 로 간주됨/
parser.add_argument('--ld', default=2, type=float, help='weight for inverted loss')  # Loss inversion 시, inversion된 loss 의 weight
parser.add_argument('--model_name', default='RESNET50')  # 사용할 모델 선택
parser.add_argument('--exp', type=str, help='model explanation', required=True)  # 훈련 방식 메모
parser.add_argument('--resize', default=448, type=int)  # 이미지 크기 재설정
parser.add_argument('--num_workers', default=6, type=int)  # 훈련에 사용할 CPU 코어 수
parser.add_argument('--esct', type=int, default=20, help='early stop count')  # early stopping 기준
parser.add_argument('--epochs', default=11, type=int)  # 전체 훈련 epoch
parser.add_argument('--batch_size', default=16, type=int)  # 배치 사이즈
parser.add_argument('--lr', '--learning_rate', default=0.0001, type=float)  # learning rate
parser.add_argument('--momentum', default=0.9, type=float)  # optimizer의 momentum
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float)  # 가중치 정규화
parser.add_argument('--optim', default='SGD')  # optimizer
global args
args=parser.parse_args()

#### make folders
if not os.path.exists('./log'):
    os.mkdir('./log')
if not os.path.exists('./model_weight'):
    os.mkdir('./model_weight')
if not os.path.exists('./checkpoint'):
    os.mkdir('./checkpoint')

global device
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

def log(message):
    with open(osp.join(args.log_path, args.model_name)+'_'+f"{args.exp}"+'.txt', 'a+') as logger:
        logger.write(f'{message}\n')

import Utility_tuning as U        

def main_worker(args): 
    print('Start Setting')
    ### 기본 setting!
    global models
    models=U.Create_Models(base_model=args.model_name, device=device)
    global criterions
    criterions = {
    "Reconstruction": nn.MSELoss(reduction='none'),
    }   
    global optimizers
    if args.optim=='SGD':
        optimizers = {
            # Encoder
            "Encoder": optim.SGD(models["Encoder"].parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4),
            # Encoder with regularization for Generator loss
            "Encoder_reg": optim.SGD(models["Encoder"].parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4),
            # Decoder
            "Decoder": optim.SGD(models["Decoder"].parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4),
            # Discriminator
            "Discriminator": optim.SGD(models["Discriminator"].parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4),
        }
    else:             
        optimizers = {
            # Encoder
            "Encoder": optim.Adam(models["Encoder"].parameters(), lr=0.0001, betas=(0.5, 0.9)),
            # Encoder with regularization for Generator loss
            "Encoder_reg": optim.Adam(models["Encoder"].parameters(), lr=0.0001),
            # Decoder
            "Decoder": optim.Adam(models["Decoder"].parameters(), lr=0.0001, betas=(0.5, 0.9)),
            # Discriminator
            "Discriminator": optim.Adam(models["Discriminator"].parameters(), lr=0.0001, betas=(0.5, 0.9)) 
        }        
    
    global encoder_scheduler
    global decoder_scheduler
    global discriminator_scheduler
    global generator_schedular
    
    if args.optim=='SGD':
        encoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizers["Encoder"], patience=5)
        decoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizers["Decoder"], patience=5)
        discriminator_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizers["Discriminator"], patience=5)
        generator_schedular = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizers["Encoder_reg"], patience=5)
        
    args.start_epoch=0
    # train transforms
    train_compose=A.Compose([
        A.Resize(args.resize, args.resize),
        A.HorizontalFlip(p=0.4),
        A.VerticalFlip(p=0.4),
        A.RandomRotate90(p=0.4),
        A.RandomGridShuffle(p=0.4),
        ToTensorV2()
    ])
    # validation transforms
    valid_compose=A.Compose([
        A.Resize(args.resize,args.resize),
        ToTensorV2()    
    ])
    
    # dataset
    print('making dataset')
    
    global train_dataset
    global valid_dataset
    global test_dataset
    
    base_root='/home/test/hdd/Dataset/chicken/Dataset/preprocessed/classification/' # with oversampling
    #base_root='/home/test/hdd/Dataset/chicken/Dataset/preprocessed2/classification/' # without oversampling
    train_dataset=dataset_.ChickenDataset(base_PATH=osp.join(base_root, 'train'), transforms=train_compose)
    valid_dataset=dataset_.ChickenDataset(base_PATH=osp.join(base_root, 'val'), transforms=valid_compose)
    test_dataset=dataset_.ChickenDataset(base_PATH=osp.join(base_root, 'test'), transforms=valid_compose)
    
    # dataloader
    global train_loader
    global val_loader
    global test_loader
    
    train_loader=torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    val_loader=torch.utils.data.DataLoader(
    valid_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)
    test_loader=torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)
    
    print('Start Tuning')
    ### 파라미터 튜닝 시작!
    pbounds = {
                'mu' : (0.1,0.5),
                'sigma': (0.2,1.0)
                }
    from bayes_opt import BayesianOptimization
    import numpy as np
    print("Let's Go!")
    from_=time.time()
    bo=BayesianOptimization(f=parameter_tuning, pbounds=pbounds, verbose=2, random_state=1 )  
    
    bo.maximize(init_points=2, n_iter=2, acq='ei', xi=0.01)

    print(bo.max)
    to_=time.time()
    log(f'\nTotal Running Time: {int((to_-from_)//60)}m {int((to_-from_)%60)}s')

if __name__=='__main__':
    main_worker(args)