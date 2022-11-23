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


def obtain_init_pool(args):
    '''
    Go to the dataset root. Get train.csv
    shuffle train.csv and get the first "init_size" samples.
    create three new csv files -> init_pool.csv, labeled.csv and unlabeled.csv
    '''
    init_pool_size = args.init_size

    original_file = os.path.join(args.dataset_root, 'original.csv')

    labeled_file = os.path.join(args.dataset_root, 'labeled.csv')
    unlabeled_file = os.path.join(args.dataset_root, 'unlabeled.csv')

    # trainfile(주소 + 파일명)의 정보를 불러와라. 
    original_rows = np.genfromtxt(original_file, delimiter=',', dtype=str)

    np.random.shuffle(original_rows)

    # 여기서의 Labeled data / Unlabeled 데이터는 개수만 나눠준다. 
    # 현재 init pool size =0 임. 추후 sampling에 따라서 추가할 예정
    # 현재 unlabeled_row에도 label이 추가되어 있는 상황. 나중에 정보를 읽을 때 sample['label'] 이 형태로 label을 불러올 예정. 
    labeled_rows = original_rows[:init_pool_size]
    unlabeled_rows = original_rows[init_pool_size:]
 
    # labeled_file(주소 + 파일명)에 Llabeled_rows ndarray 정보를 저장해라 
    # dataset_root 위치에다가, labeled.csv / unlabeled.csv 로 저장해라. 
    np.savetxt(labeled_file, labeled_rows,'%s',delimiter=',', encoding='utf-8')
    np.savetxt(unlabeled_file, unlabeled_rows,'%s',delimiter=',', encoding='utf-8')

    return labeled_file, unlabeled_file
