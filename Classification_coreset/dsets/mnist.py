from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import csv 

import pdb
#    dataset_test = MNIST(args.dataset_root, subset='test', csv_file='test.csv', transform=data_transforms)
class MNIST(Dataset):

    def __init__(self, root_dir, subset, csv_file, transform=None):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir,'images')
        
        if '/' not in csv_file:
            self.dataframe = pd.read_csv(os.path.join(root_dir,csv_file), header=None)
        else:
            self.dataframe = pd.read_csv(csv_file, header=None)
        self.transform = transform

        self.subset = subset # train or test

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        #데이터 셋은 image의 이름 / Label 정보 2개의 열을 가지고 있는 csv 파일과, 
        # 각각의 subset(train, test) 폴더 안에 각 이미지들이 있어야 한다. 

        img_name = os.path.join(self.root_dir, self.subset, self.dataframe.iloc[idx,0])
        img_name_small = self.dataframe.iloc[idx, 0]

        # image 자체를 저장하는 것. 그럼 csv를 받아들여야 하는게 아니라, image로 csv 파일로 만들어야 하는건가? 
        # io.imread 는 이미지 파일을 불러온다. 

        # io.imread는 nd.array 형식으로 결과값을 반환함. 
        image = io.imread(img_name)

        label = self.dataframe.iloc[idx,1]
        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'label': label, 'img_name': img_name_small}

        return sample
