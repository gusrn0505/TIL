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
from datetime import datetime
import argparse
import pprint

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# local stuff
from dsets.mnist import MNIST
from mymodels.mnist_net import Net
from train_test import train, test

def obtain_init_pool(args):
    '''
    Go to the dataset root. Get train.csv
    shuffle train.csv and get the first "init_size" samples.
    create three new csv files -> init_pool.csv, labeled.csv and unlabeled.csv
    '''
    init_pool_size = args.init_size

    train_file = os.path.join(args.dataset_root, 'train.csv')
    init_file = os.path.join(args.dataset_root, 'init_pool.csv')
    labeled_file = os.path.join(args.dataset_root, 'labeled.csv')
    unlabeled_file = os.path.join(args.dataset_root, 'unlabeled.csv')

    # trainfile(주소 + 파일명)의 정보를 불러와라. 
    train_rows = np.genfromtxt(train_file, delimiter=',', dtype=str)

    np.random.shuffle(train_rows)

    # 여기서의 Labeled data / Unlabeled 데이터는 개수만 나눠준다. 
    # 그럼 여기서 데이터가 Label을 가지고 있어야 하는지 유무는 어떻게 체크할 수 있지? 
    # 모델에서 Label 데이터를 어떻게 받아들이는지 확인해야겠네. 
    labeled_rows = train_rows[:init_pool_size]
    unlabeled_rows = train_rows[init_pool_size:]
    print(labeled_rows.shape)
    print(unlabeled_rows.shape)


    # 이게 양식이 잘못 됐다고? 
    # 숫자값을 string으로 인식할 줄 알고 이렇게 한 건가? 
    # 일단 $s,$s 을 $f, $f로 변경해보겠음 
    # 뜨아.. $s, $s 가 아니라 $s 하나만 넣는 거였다. 

    # labeled_file(주소 + 파일명)에 Llabeled_rows ndarray 정보를 저장해라 
    np.savetxt(labeled_file, labeled_rows,'%s',delimiter=',')
    np.savetxt(init_file, labeled_rows,'%s',delimiter=',')
    np.savetxt(unlabeled_file, unlabeled_rows,'%s',delimiter=',')


#    np.savetxt(labeled_file, labeled_rows,'%s,%s',delimiter=',')
#    np.savetxt(init_file, labeled_rows,'%s,%s',delimiter=',')
#    np.savetxt(unlabeled_file, unlabeled_rows,'%s,%s',delimiter=',')

    return labeled_file, unlabeled_file
