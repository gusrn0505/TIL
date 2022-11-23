from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt

# DataLoader은 Dataset을 샘플에 쉽게 접근할 수 있도록 순회가능한 객체(iterable)로 감쌉니다
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.models as models 
import pdb
from datetime import datetime
import argparse
# pretty print. 들여쓰기 등을 지원해준다. 
import pprint
import time
import csv

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# local stuff
# 폴더에 있는 경우 A.B 형태로 기술 
from dsets.mnist import MNIST
from mymodels.mnist_net import AutoEncoder
from train_test import train, test
from init_pool_tools import obtain_init_pool
from coreset import Coreset_Greedy

# 옵션에 따라 파이썬 스크립트가 다르게 동작하도록 해주기 위한 명령어 - argparser
# Parsing 이란 어떤 페이지(문서, HTML 등)에서 내가 원하는 데이터를 특정 패턴이나 순서로 추출하여 정보로 가공하는 것을 의미 

# 아래 내용 autoencoder에 맞게 수정필요! 
def argparser():
    parser = argparse.ArgumentParser(description='Active Learning - Image Classification')
    # 파싱할 인자들 추가하기

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

# batch size 수정 필요 
    parser.add_argument('--al-batch-size', default=500, type=int,
                        help='number of samples to add in each iteration')

# init size 수정 필요  
    parser.add_argument('--init-size', default=1000, type=int,
                        help='init pool size')

    # Dataset의 위치는 data\mnist_easy 에 있는 걸 default 로 여김 
    # 해당 위치에 csv 타입의 Dataset을 준비할 필요가 있음. 
    # arg.dataset_root 를 하면 따로 값을 설정하지 않는 한 default 값이 나온다. 
    parser.add_argument('--dataset-root', default='data/mnist_easy', type=str,
                        help='root directory of the dataset')
    parser.add_argument('--dataset-name', default='mnist', type=str,
                        help='dataset name')
    parser.add_argument('--output-dir', default='output', type=str,
                        help='dataset name')

# MNIST 용으로 하기 때문에 clas는 10개로 
    parser.add_argument('--nclasses', type=int, default=10, metavar='N',
                        help='number of classes in the dataset')


# Sampling method는 coreset 하나다! 
    parser.add_argument('--sampling-method', default='random', type=str,
                        help='one of random, coreset')

    parser.add_argument('--max-eps', type=int, default=10, metavar='N',
                        help='max episodes of active learning')
    # bald method 가 의미하는 건 뭘까? Sampling 방식 중에 Bald method 라는 게 있음. 

    parser.add_argument('--dropout-iterations', type=int, default=5, metavar='N',
                        help='dropout iterations for bald method')

    # batch-size 인자 추가하기. type = int, default =32, 인자의 이름은 "N", -h로 인자에 대해 찾으면 아래 메시지가 나옴
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')

    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')

    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')


######### 아래 코드들은 이번 Coreset selection 간 사용하지 않을 값들 ##################
#    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                        help='how many batches to wait before logging training status')



    return parser


# Unlabel dataset에서 Coreset selection을 통해서 Sampling 한 값들 없애기
def remove_rows(unlabel_dataset, sampling_dataset):

    len_unlabel = len(unlabel_dataset)
    len_sampling = len(sampling_dataset)

    # .tolist() 는 어떤 함수일까? 
    # numpy, pandas 데이터에 대해서 같은 레벨(위치 - index)에 있는 데이터끼리 묶어준다. 
    # 각 데이터의 특성값들을 한 list type으로 변경 
    unlabel_dataset = unlabel_dataset.tolist()
    sampling_dataset = sampling_dataset.tolist()

    # perm에 있는 데이터 중 samp에 속하지 않은 데이터 뽑기 
    # X/s 구만. Unlabelled dataset 
    result = [item for item in unlabel_dataset if item not in sampling_dataset]

    # 혹시나 samp에 속한 data를 뽑거나, 중복으로 뽑은 경우 오류 발생 
    assert len(result) == len_unlabel - len_sampling
    return np.array(result)


# get_feature 부분을 auto encoder을 통한 latent variable 확보로 바꿔내야 함. 


# 모델 학습 간 loader에 있는 정보들을 모델에 입력할 수 있도록 구분시켜 준 후, 
# GPU에서 model.get_features 시행. 이후에 CPU 메모리로 돌려서 featrue 반환

# Model을 Net에서 Autoencoder로 바꿀 것.
def get_features(model, loader):
    features = []

    # 모델 evaluation 과정에서 사용하지 않아야 하는 layer들을 알아서 off 시키는 함수 
    # model을 evaluate를 할 수 있도록 세팅시키는 것 
    model.eval()

    count = 0
    # torch.no_grad : autograd engine을 비활성화시켜 필요한 메모리를 줄여주고 연산속도를 증가시킴 
    with torch.no_grad():
        for sample in loader:
            # sample의 feature들을 각각 변수(data, target 등)으로 분리하기  
            # data는 nd.array 형식임. 
            data = sample['image']
            target = sample['label']
            img_name = sample['img_name'][0]

            # data.to(device) : GPU에 data의 복사본이 반환됨
            data, target = data.to(device), target.to(device)

# model.get_features => model.get_codes 
            output = model.get_codes(data)
            ## pdb.set_trace()

            count += 1
            # if count > 10000:
            #     break

            # .cpu() : GPU 메모리에 올려져 있는 tensor을 cpu 메모리로 복사하는 method
            # .numpy() : tensor을 numpy로 반환하여 반환. Cpu에 있는 tensor에 대해서만 사용 가능
            features.append(output.cpu().numpy())
            ## features.append((img_name, output.cpu().numpy()))
    return features

"""
# uncertainty 등 우선순위를 비교할 수 있는 기준을 계산하는 것 
def get_probs(model, loader, stochastic=False):
    probs = []
    if stochastic:
        # 모델을 Train mode로 변경. Stochasitic 이면 다르게 해야할 이유가 있나? 
        model.train()
    else:
        model.eval()

    count = 0
    with torch.no_grad():
        for sample in loader:
            # data : ndarray 형식
            data = sample['image'] 
            target = sample['label']
            img_name = sample['img_name'][0]

            data, target = data.to(device), target.to(device)

            # Stochastic일 경우 model의 stocastic_pred 함수를 실행 
            if stochastic:
                output = model.stochastic_pred(data)
            
            # model 에 그냥 data를 입력하는 것과, 위의 stochastic 일 때 랑 어떤 차이가 있나? 
            output = model(data)

            # convert log softmax into softmax outputs
            prob = output.cpu().numpy()
            # output의 [0]의 값이 의미하는 게 뭘까? 
            prob = np.exp(prob[0])

            probs.append(prob)

            count += 1 # 엥 count가 역할을 하는 게 없네. 

    return np.array(probs)


# torch.utils.data.Dataset : 샘플과 정답(label)을 저장한다. 
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

    # 주어진 인덱스(idx) 에 해당하는 샘플을 데이터셋에서 불러오고 반환함.
    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.subset, self.dataframe.iloc[idx,0])
        img_name_small = self.dataframe.iloc[idx, 0]
        image = io.imread(img_name)

        label = self.dataframe.iloc[idx,1] # 데이터 셋에서 인덱스가 1이면 Label이 아닌 것 같은데..?
        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'label': label, 'img_name': img_name_small}

        return sample
"""

# uncertainty, margin, coreset 등 다양한 방식으로 sampling 하기 
def active_sample(args, unlabeled_rows, sample_size, method='coreset', model=None):
    if method == 'random':
        # 아에 list의 순서를 바꿔서 한번에 뽑는구나. 이게 계산양이 적겠다. 
        np.random.shuffle(unlabeled_rows)
        sample_rows = unlabeled_rows[:sample_size]
        return sample_rows
    
    if method == 'coreset':
        #create unlabeled loader
        data_transforms = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])

        unlab_dset = MNIST(args.dataset_root, subset='train',csv_file='unlabeled.csv',transform=data_transforms)
        unlab_loader = DataLoader(unlab_dset, batch_size=1, shuffle=False, **kwargs)

        #labeled dataloader
        lab_dset = MNIST(args.dataset_root, subset='train',csv_file='labeled.csv',transform=data_transforms)
        lab_loader = DataLoader(lab_dset, batch_size=1, shuffle=False, **kwargs)

        # get labeled features. Autoencoder로부터 latent variable 산출하기 
        labeled_features = get_features(model, lab_loader) # (img_name, features)
        # get unlabeled features
        unlabeled_features = get_features(model, unlab_loader)# (img_name, features)

        all_features = labeled_features + unlabeled_features
        labeled_indices = np.arange(0,len(labeled_features))

        # Coreset_Greedy 함수 수정 필요! 결과값을 다르게 뽑아야 함!
        coreset = Coreset_Greedy(all_features)

        # unlabeled 데이터에서 sample_size 만큼 center point 뽑기 
        new_batch, max_distance = coreset.sample(labeled_indices, sample_size)
        
        # unlabeled rows start after labeled rows in all_features
        # so offset the indices
        new_batch = [i - len(labeled_features) for i in new_batch]
        
        sample_rows = unlabeled_rows[new_batch]

        return sample_rows
    


# dest_dir 위치에 결과 기록하기. 입력할 결과들을 입력값으로 넣음 
def log(dest_dir, episode_id, sample_method, sample_time, accuracy, labeled_rows):
    # log file은 dest_dir 위치에 log.csv를 두기 위한 주소이다. 
    log_file = os.path.join(dest_dir, 'log.csv')

    # 주소가 정확하지 않을  해당 위치에 파일이 존재하지 않을 때, log_rows를 다음과 같이 정한다. 
    if not os.path.exists(log_file):
        log_rows = [['Episode Id','Sample Method','Sampling Time (s)','Labeled Pool','Accuracy']]
    # 파일이 존재할 때에는 데이터를 처리해서 불러온다. 
    else:
        log_rows = np.genfromtxt(log_file, delimiter=',', dtype=str).tolist()

    # episod_id, sample_mthod, sample_time 등의 값을 추가한다. 
    log_rows.append([episode_id,sample_method, sample_time, len(labeled_rows), accuracy])
    
    # 데이터를 저장한다. 파일이 없다면 새로 만든다. 
    np.savetxt(log_file,log_rows,'%s,%s,%s,%s,%s',delimiter=',')


def log_picked_samples(dest_dir, samples, ep_id=0):
    dest_file = os.path.join(dest_dir, 'picked.txt')

    # 기존 파일에 추가모드. 
    with open(dest_file, 'a') as f:
        #csvfile에 대해서 바꿀 수 있도록 설정  
        writer = csv.writer(f)
        # ["Episode ID", str(ep_id)] (가장 첫줄) 의 값을 한 줄씩 저장하기. 
        writer.writerow(["Episode ID", str(ep_id)])
        for s in samples:
            # 그 다음으로 sample들의 feature들을 한 객체로 뭉쳐 추가해주기. 
            writer.writerow(s.tolist())

# original 데이터를 구분해주기 위해서 함수 추가 
def permutation_train_test_split(X, test_size=0.2, shuffle=True, random_state = 1004) : 
    test_num = int(X.shape[0] * test_size)
    train_num = X.shape[0] - test_num 

    if shuffle : 
        np.random.seed(random_state)
        shuffled = np.random.permutation(X.shape[0])
        X = X[shuffled, :]
        X_train = X[:train_num]
        X_test = X[train_num :]

    else : 
        X_train = X[:train_num]
        X_test = X[train_num :]

    return X_train, X_train

"""
# args 로 Dataset을 구분하여 저장한 다음, 그때 그때 필요한 값을 뽑아 쓰는 건 좋은 것 같다. 
# 이 함수를 통해서 필요로 하는 데이터 셋들을 구분하는 거구나. 
# train.csv 에 걍 더 지금 가지고 있는 csv을 넣으면 되는 건가? 
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

    train_rows = np.genfromtxt(train_file, delimiter=',', dtype=str)

    np.random.shuffle(train_rows)

    labeled_rows = train_rows[:init_pool_size]
    unlabeled_rows = train_rows[init_pool_size:]

    np.savetxt(labeled_file, labeled_rows,'%s,%s',delimiter=',')
    np.savetxt(init_file, labeled_rows,'%s,%s',delimiter=',')
    np.savetxt(unlabeled_file, unlabeled_rows,'%s,%s',delimiter=',')

    return labeled_file, unlabeled_file
"""

"""
# 모델을 학습할 때 필요한 과정들을 압축시켜 놓은 함수 
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, sample in enumerate(train_loader):
        data = sample['image']
        target = sample['label']

        data, target = data.to(device), target.to(device)
        # pdb.set_trace()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return model


    # 여기에 원 데이터를 넣으면 train.csv, test.csv 로 나눠줄 수 있도록 해주자. 
    from sklearn.model_selection import train_test_split



"""
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

    # Obtaining init pool - 추가 : Original.csv가 있다면 이를 Train과 Test로 구분해줌. 
    # 현재 코드에서 Train set 에는 Label이 달려 있어야 하나? ㅇㅇ 하나의 파일에서 Label과 데이터를 구분하고 있음. 
    original_file = os.path.join(args.dataset_root, 'original.csv')
    original_rows = np.genfromtxt(original_file, delimiter=',', dtype=str)
    train_original, test_original = permutation_train_test_split(original_rows, test_size=0.2, shuffle = True)

    origin_train_file = os.path.join(args.dataset_root, 'train.csv')
    origin_test_file = os.path.join(args.dataset_root, 'test.csv')
    
    np.savetxt(origin_train_file, train_original,'%s',delimiter=',')
    np.savetxt(origin_test_file, test_original,'%s',delimiter=',')

    # 여기서 csv 파일을 먼저 받는데? 
    labeled_csv, unlabeled_csv = obtain_init_pool(args)
    print("Initial labeled pool created.")

    # initial setup
    # data transform을 어떻게 한다는 걸까. csv 파일을 넣으면 어떻게 되지? 
    data_transforms = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])

    # 데이터를 위치 등을 정리 및 sample = {'image': image, 'label': label, 'img_name': img_name_small} 형태로 수정해줌
    dataset_train = MNIST(args.dataset_root, subset='train',csv_file='labeled.csv',transform=data_transforms)
    dataset_test = MNIST(args.dataset_root, subset='test', csv_file='test.csv', transform=data_transforms)
    dataset_trial = MNIST(args.dataset_root, subset = 'trial',csv_file = 'init_pool.csv', transform=data_transforms)
    print("len :", len(dataset_trial))
    print("indexing :", dataset_trial[0],dataset_trial[1] )

    # img_name = os.path.join(self.root_dir, self.subset, self.dataframe.iloc[idx,0])

    # initial training

    trial_loader = DataLoader(dataset_trial, batch_size =2, shuffle=True, **kwargs)
    # 왜 Dataloader의 정보를 못 불러오는 걸까? 다른 예시에서는 잘 불러오는데? MNIST의 자료 정보가 이상한거라면 Dataloader 자체는 잘 실행이 되는게 이상하고
    trial = iter(trial_loader)
    # 이것도 안 되네. 왜 이번엔 print에서 터질까. 
    print(next(trial)) 



    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(dataset_test, batch_size=args.test_batch_size, shuffle=False, **kwargs)


#    model = Net().to(device) # initialize the model.
# Net 모델을 Autoencoder로 변경해줌. 변경해줌에 따라서 아래 코드를 변경해줘야할 수 있음. 
    model = AutoEncoder().to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr) # setup the optimizer
    # 학습률을 각 Step 마다 조절해주는 함수 
    scheduler = StepLR(optimizer, step_size = 1, gamma=args.gamma)


    for epoch in range(1, args.epochs + 1):
        # train_test.py 의 train 함수를 통해 손 쉽게 모델 학습 진행 
        model = train(args, model, device, train_loader, optimizer, epoch)
        
        scheduler.step()

        # train_test.py 의 test 함수를 통해 손 쉽게 모델 학습 진행 
    accuracy = test(args, model, device, test_loader)

    # save model
    dest_dir = os.path.join(args.output_dir, args.dataset_name)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    now = datetime.now()
    dest_dir_name = str(now.year) + str(now.month) + str(now.day) + str(now.hour) + str(now.minute) + str(now.second)
    dest_dir_name = os.path.join(dest_dir, dest_dir_name)
    if not os.path.exists(dest_dir_name):
        os.mkdir(dest_dir_name)
    save_path = os.path.join(dest_dir_name,'init.pth')

    
    torch.save(model.state_dict(), save_path)
    print("initial pool model saved in: ",save_path)



    # copy labeled csv and unlabeled csv to dest_dir
    # pdb.set_trace()

    # save config
    with open(dest_dir_name + '/config.json', 'w') as f:
        import json
        # 파이썬 객체를 JSON 파일로 저장하기. 
        json.dump(vars(args),f)
    # save logs

    # pdb.set_trace()
    # 기록 남기기
    log(dest_dir_name, 0, args.sampling_method, 0, accuracy, [0]*args.init_size)
    log_picked_samples(dest_dir_name, np.genfromtxt(labeled_csv, delimiter=',', dtype=str))


    # start the active learning loop.
    episode_id = 1
    while True:

        if episode_id > args.max_eps:
            break


        # read the unlabeled file
        unlabeled_rows = np.genfromtxt(unlabeled_csv, delimiter=',', dtype=str)
        labeled_rows = np.genfromtxt(labeled_csv, delimiter=',', dtype=str)

        print("Episode #",episode_id)


        # sanity checks
        # 더 이상 남은 unlabled row가 없을 때 끝내기 
        if len(unlabeled_rows) == 0:
            break

        # set the sample size
        sample_size = args.al_batch_size
        if len(unlabeled_rows) < sample_size:
            sample_size = len(unlabeled_rows)

        # sample
        sample_start = time.time()
        sample_rows = active_sample(args, unlabeled_rows, sample_size, method=args.sampling_method, model=model)
        sample_end = time.time()

        # log picked samples
        log_picked_samples(dest_dir_name, sample_rows, episode_id)



        sample_time = sample_end - sample_start

        # update the labeled pool
        labeled_rows = np.concatenate((labeled_rows,sample_rows),axis=0)
        np.savetxt(labeled_csv, labeled_rows,'%s,%s',delimiter=',')


        # update the unlabeled pool
        unlabeled_rows = remove_rows(unlabeled_rows, sample_rows)
        np.savetxt(unlabeled_csv, unlabeled_rows, '%s,%s', delimiter=',')

        print("Unlabeled pool size: ",len(unlabeled_rows))
        print("Labeled pool size: ",len(labeled_rows))


        #train the model
        dataset_train = MNIST(args.dataset_root, subset='train',csv_file='labeled.csv',transform=data_transforms)
        train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, **kwargs)
  

        model = Net().to(device) # initialize the model.
        optimizer = optim.Adam(model.parameters(), lr=args.lr) # setup the optimizer
        # scheduler = StepLR(optimizer, step_size = 1, gamma=args.gamma)

        for epoch in range(1, args.epochs + 1):
            model = train(args, model, device, train_loader, optimizer, epoch)
        accuracy = test(args, model, device, test_loader)
            # scheduler.step()

        # save model
        save_path = os.path.join(dest_dir_name, 'ep_'+str(episode_id)+'.pth')
        torch.save(model.state_dict(), save_path)

        log(dest_dir_name, episode_id, args.sampling_method, sample_time, accuracy, labeled_rows)

        episode_id += 1
