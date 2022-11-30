from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt

# DataLoader은 Dataset을 샘플에 쉽게 접근할 수 있도록 순회가능한 객체(iterable)로 감쌉니다
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
from torchvision.transforms import ToTensor
import torchvision.models as models 
from collections import defaultdict

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
from sklearn.metrics import pairwise_distances

# local stuff
# 폴더에 있는 경우 A.B 형태로 기술 
from dsets.mnist import MNIST
from mymodels.mnist_net import Net
from auto_encoder import AutoEncoder, ConvAutoEncoder, ae_train
from train_test import train, test
from init_pool_tools import obtain_init_pool
from coreset import Coreset_Greedy




def argparser():
    parser = argparse.ArgumentParser(description='Active Learning - Image Classification')
    # 파싱할 인자들 추가하기

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

# batch size 수정 필요 
    parser.add_argument('--al-batch-size', default=100, type=int,
                        help='number of samples to add in each iteration')

# init size 수정 필요. Batch size와 동일하게 
    parser.add_argument('--init-size', default=0, type=int,
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
    parser.add_argument('--sampling-method', default='coreset', type=str,
                        help='one of random, coreset')

    parser.add_argument('--max-eps', type=int, default=1, metavar='N',
                        help='max episodes of active learning')


    parser.add_argument('--dropout-iterations', type=int, default=5, metavar='N',
                        help='dropout iterations for bald method')

    parser.add_argument('--dim-reduction', type=str, default="AE", metavar='LR',
                        help='method of dimension reduction')

    parser.add_argument('--hidden-dim1', type=int, default=64, metavar='N',
                        help='first hidden dimension of AE')

    parser.add_argument('--hidden-dim2', type=int, default=32, metavar='N',
                        help='second hidden dimension of AE')

    # batch-size 인자 추가하기. type = int, default =32, 인자의 이름은 "N", -h로 인자에 대해 찾으면 아래 메시지가 나옴
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')

    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')

    parser.add_argument('--ae-epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train autoencoder (default: 50)')


    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')


    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')



    return parser


# Unlabel dataset에서 Coreset selection을 통해서 Sampling 한 값들 없애기
def remove_rows(unlabel_dataset, sampling_dataset):

    unlabel_dataset = np.array(unlabel_dataset).tolist()
    sampling_dataset = np.array(sampling_dataset).tolist()

    result = [item for item in unlabel_dataset if item not in sampling_dataset]

    # 혹시나 samp에 속한 data를 뽑거나, 중복으로 뽑은 경우 오류 발생. 아마 reconstruction 중에 중복 값이 생길 수 있음. 
    # assert len(result) == len_unlabel - len_sampling
    return np.array(result)


def get_features(model, dataset, device):
    features = []

    # 모델 evaluation 과정에서 사용하지 않아야 하는 layer들을 알아서 off 시키는 함수 
    # model을 evaluate를 할 수 있도록 세팅시키는 것 
    model.eval()
    # torch.no_grad : autograd engine을 비활성화시켜 필요한 메모리를 줄여주고 연산속도를 증가시킴 
    
    with torch.no_grad() : 
        for sample in dataset:
            sample = torch.tensor(sample, requires_grad=True).to(device)
            # data는 nd.array 형식임. 
        
            output = model.get_codes(sample)

            features.append(output.cpu().numpy())
            ## features.append((img_name, output.cpu().numpy()))
    return features


# uncertainty, margin, coreset 등 다양한 방식으로 sampling 하기 
def active_sample(unlabeled_dataset, labeled_dataset, sample_size, method='ae_coreset', model=None, device="cpu"):
    if method == 'ae_coreset':

        labeled_features = get_features(model, labeled_dataset, device) # (img_name, features)
        unlabeled_features = get_features(model, unlabeled_dataset, device)# (img_name, features)


        all_features = labeled_features + unlabeled_features
        # label data의 index가 어디까지인지 표기. 
        labeled_indices = np.arange(0,len(labeled_features))

        #coreset = Coreset_Greedy(all_features)
        coreset = Coreset_Greedy(all_features, len(labeled_features))

        # unlabeled 데이터에서 sample_size 만큼 center point 뽑기, 당시 반지름 뽑기
        new_batch, max_distance = coreset.sample(labeled_indices, sample_size)
        
        # unlabeled rows start after labeled rows in all_features
        # so offset the indices
        new_batch = [i - len(labeled_features) for i in new_batch]
        new_batch.sort()

        sample_rows = [unlabeled_dataset[i] for i in new_batch]
        return sample_rows, new_batch, max_distance


# 수정 필요. 한번 Unlabeled data가 일부 데이터를 뺸 이후에선 Original dataset 과의 Index가 맞지 않음 
def make_subgraph(sampling_index, original_dataset, radii, model):
# original_dataset을 Label data로 바꿀 순 없나? 
    # x는 Sample point 
    x = [original_dataset[i] for i in sampling_index]
    dataset = original_dataset

    if model is not None : 
        x = get_features(model, x, device)
        dataset = get_features(model, dataset, device)

    dist = pairwise_distances(x,dataset, metric='euclidean')

    subgraph= dist.copy()
    density_subgraph = []
    for i, row in enumerate(dist) : 
        for j, distance in enumerate(row) : 
            if distance > radii or j == sampling_index[i] : subgraph[i,j] =int(0) # 자기자신은 제거  
            else : subgraph[i,j] = int(1) 
        
        density_subgraph.append(sum(subgraph[i]))
    

    return np.array(subgraph), density_subgraph


# 추가 수정 필요! 뭔가 이상함!
def adjacency_subgraph(labeled_dataset, labeled_dataset_label, radii, model) : 
    dataset = labeled_dataset
    num_subgraph = len(labeled_dataset_label)
    if model is not None :
        dataset = get_features(model,dataset, device)
    
    dist = pairwise_distances(dataset, dataset, metric='euclidean')
    adj_dist = dist.copy()
    
    for i, row in enumerate(dist) : 
        adj_d = np.where(row < 2*radii[0])[0] 
        for j, distance in enumerate(row) : 
            if distance >= 2*radii[0] : adj_dist[i,j] = int(0)   
            # 뭔가 이상.. 여기선 문제없지만, 밖에서 다시 거리를 재면 성립안하는 경우가 있음. 
            elif 2*radii[0] > distance and distance >= radii[0]  : 
                adj_dist[i,j] = int(1)
            elif i==j : adj_dist[i,j] = int(0) # 자기자신은 제거
            else : 
                print('Break')

    classified_subgraph_index = []

    for i in range(num_subgraph) : 
        i_sub_class = "x"
        adj_index = np.where(adj_dist[ :,i] ==1)
        if len(adj_index[0])==0 : continue 
        i_sub_class = labeled_dataset_label[i]

        for j in adj_index[0] :  
            if i_sub_class != labeled_dataset_label[j] : 
                i_sub_class = "x"
                continue
        if i_sub_class != "x" : 
            classified_subgraph_index.append(i)
    
    classified_label = [labeled_dataset_label[i] for i in classified_subgraph_index]

    return dist, adj_dist, classified_subgraph_index, classified_label



def first_classification(classified_subgraph_index, pseudo_class_label, subgraph, density_subgraph, ratio) : 
    dense_classified_subgraph_index = [density_subgraph[i] for i in classified_subgraph_index]
    sorted_density = sorted(dense_classified_subgraph_index, reverse=True)
    rank = int(ratio*len(sorted_density))

    M = sorted_density[max(rank-1, 0)]

    classification = defaultdict(list)
    for i, index in enumerate(classified_subgraph_index) : 
        if density_subgraph[index] < M : continue
        x_index = list(np.where(subgraph[index] == 1)[0])
        label = pseudo_class_label[i]
    
        classification[label] = classification[label] + x_index
    return classification

def check_performance(classification, original_label) : 
    score = defaultdict(list) 
    all_score = 0 
    all_count = 0 
    for i in sorted(list(classification.keys())) : 
        x_index = classification[i] 
        num_x = len(x_index)
        count = 0 
        for index in x_index :
            if original_label[index] == i : count += 1 
        
        i_score = count/num_x
        all_score += count
        all_count += num_x
        score[i] = [num_x, i_score]
    
    all_score = all_score/all_count
    
    return all_count, all_score, score



# dest_dir 위치에 결과 기록하기. 입력할 결과들을 입력값으로 넣음 
def log(dest_dir, episode_id, sample_method,  label_dataset_label, num_classification, ratio, accuracy):
    # log file은 dest_dir 위치에 log.csv를 두기 위한 주소이다. 
    log_file = os.path.join(dest_dir, 'log.csv')

    # 주소가 정확하지 않을  해당 위치에 파일이 존재하지 않을 때, log_rows를 다음과 같이 정한다. 
    if not os.path.exists(log_file):
        log_rows = [['Episode Id','Sample Method','Labeled Pool', 'Num of classification', 'Ratio', 'Accuracy']]
    # 파일이 존재할 때에는 데이터를 처리해서 불러온다. 
    else:
        log_rows = np.genfromtxt(log_file, delimiter=',', dtype=str, encoding='utf-8').tolist()

    # episod_id, sample_mthod, sample_time 등의 값을 추가한다. 
    log_rows.append([episode_id,sample_method, len(label_dataset_label), num_classification, ratio, accuracy])
    
    # 데이터를 저장한다. 파일이 없다면 새로 만든다. 
    np.savetxt(log_file,log_rows,'%s,%s,%s,%s,%s,%s',delimiter=',')


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



def check_class(subgraph, density_subgraph, M, labeled_dataset_label) : 
    num_sample = np.shape(subgraph)[1]

    classification =[-1]*num_sample 
    classified_index = []

    filtered_subgraph_index = []


    # xk 가 속한 subgraph의 label이 모두 같을 확률 
    for i in range(num_sample) : 
        in_subgraph_index = np.where(subgraph[ :,i] ==1)
        if len(in_subgraph_index[0])==0 : continue 
        i_class = labeled_dataset_label[in_subgraph_index[0][0]]

        for j in in_subgraph_index[1:] :  
            if i_class != labeled_dataset_label[j] : 
                i_class = -1
                continue
        if i_class != -1 : 
            classification[i] = i_class
            classified_index.append(i)
    
    score = len(classified_index)
    pseudo_label = [classification[i] for i in classified_index]

    return score, classified_index, pseudo_label



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


    # 데이터 셋 변경 시 수정 필요 ####################################
    original_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )


    # MNIST Dataset을 가공할 수 있는 list로 변경. feature와 label 각각 저장 
    original_all = []
    original_dataset = []
    original_label = [] 

    for sample in original_data : 
        original_all.append(sample)
        feature = np.array(sample[0])
        original_dataset.append(list(feature.reshape(3*len(feature[0])*len(feature[0][0]))))
        original_label.append(sample[1])
    

    unlabeled_dataset = original_dataset[:]
    unlabeled_dataset_label = original_label[:]
    labeled_dataset = [] 
    labeled_dataset_label = []

    # 데이터 셋 변경 시 수정 필요 #####################################
    PATH = './weights/CIFAR10/'
    dim_reduction = None
    if args.dim_reduction == "AE" : 
        dim_reduction = torch.load(PATH + 'AE.pt')  
        dim_reduction.load_state_dict(torch.load(PATH + 'AE_state_dict.pt'))  

    if args.dim_reduction == "CAE" : 
        dim_reduction = torch.load(PATH + 'CAE.pt')
        dim_reduction.load_state_dict(torch.load(PATH + 'CAE_state_dict.pt'))  

    print("Successfully loaded AE & CAE")


    # set the sample size
    sample_size = args.al_batch_size
    if len(unlabeled_dataset) < sample_size:
        sample_size = len(unlabeled_dataset)

    if args.sampling_method == "ae_coreset" : 
        print(args.sampling_method)
        sample_dataset, sample_index,radius  = active_sample(unlabeled_dataset, labeled_dataset, sample_size, method=args.sampling_method, model=dim_reduction, device=device)

    if args.sampling_method == "cae_coreset" : 
        sample_dataset, sample_index,radius  = active_sample(unlabeled_dataset, labeled_dataset, sample_size, method=args.sampling_method, model=dim_reduction, device=device)

    if args.sampling_method == "coreset" : 
        sample_dataset, sample_index, radius = active_sample(unlabeled_dataset, labeled_dataset, sample_size, method=args.sampling_method, model=None, device=device)
    
    sample_data = [unlabeled_dataset[i] for i in sample_index]
    sample_label = [unlabeled_dataset_label[i] for i in sample_index]

    # Sampling에 따른 Datset 수정 
    labeled_dataset = sample_data[:]
    labeled_dataset_label = sample_label[:]

    unlabeled_dataset = remove_rows(unlabeled_dataset, sample_data)
    for i in sample_index[::-1] : 
        del unlabeled_dataset_label[i]

    print("Unlabeled pool size: ",len(unlabeled_dataset))
    print("Labeled pool size: ",len(labeled_dataset))

    subgraph, density_subgraph = make_subgraph(sample_index, original_dataset, radius, dim_reduction)
    dist_class, adj_dist, classified_subgraph_index, pseudo_class_label = adjacency_subgraph(labeled_dataset, labeled_dataset_label, radius, dim_reduction)

    print("Well work!")


    # save model
    dest_dir = os.path.join(args.output_dir, args.dataset_name)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    now = datetime.now()
    dest_dir_name = str(now.year) + str(".")+ str(now.month) + str(".") + str(now.day)+ str(".") + str(now.hour) + str(now.minute)
    dest_dir_name = os.path.join(dest_dir, dest_dir_name)

    if not os.path.exists(dest_dir_name):
        os.mkdir(dest_dir_name)


    episode_id = 0

    ratio_term = list(np.arange(0.001, 1, 0.001))
    for ratio in ratio_term : 
        f_classification = first_classification(classified_subgraph_index, pseudo_class_label, subgraph, density_subgraph, ratio)
        num_classification, score, dic_score = check_performance(f_classification,original_label)

        log(dest_dir_name, episode_id, args.sampling_method, labeled_dataset_label, num_classification, ratio, score)

    neural = Net().to(device)
    optimizer1 = optim.Adam(neural.parameters(), lr=args.lr) # setup the optimizer
    # 학습률을 각 Step 마다 조절해주는 함수 
    scheduler1 = StepLR(optimizer1, step_size = 1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        # train_test.py 의 train 함수를 통해 손 쉽게 모델 학습 진행 
        # train 수정 필요. 
        neural = train(args, neural, device, labeled_dataset, labeled_dataset_label, optimizer1, epoch)
        
        scheduler1.step()

        # train_test.py 의 test 함수를 통해 손 쉽게 모델 학습 진행 
    # 추후 pseudo labeling dataset에 대해서 바꿔줄 필요가 있음. 
    accuracy = test(args, neural, device, unlabeled_dataset, unlabeled_dataset_label, optimizer1, epoch)
    log(dest_dir_name, episode_id, "CNN", labeled_dataset_label, len(unlabeled_dataset_label), 0, accuracy)

    print("First test clear")

    """
    # start the active learning loop.
    episode_id = 2
    while True:

        if episode_id > args.max_eps:
            break

        # read the unlabeled file
        # 위 아래로 붙일 수 있도록 조정할 필요가 있음. 
        unlabeled_rows = np.genfromtxt(unlabeled_csv, delimiter=',', dtype=str, encoding='utf-8')
        labeled_rows = np.genfromtxt(labeled_csv, delimiter=',', dtype=str, encoding='utf-8')

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
        # ae 기반 sample. ae_unlabeled_rows data 형성 필요 
        if args.sampling_method == "ae_coreset" : 
            sample_rows = active_sample(args, unlabeled_rows, sample_size, method=args.sampling_method, model=model2)

        if args.smapling_method == "coreset" : 
            sample_rows = active_sample(args, unlabeled_rows, sample_size, method=args.sampling_method, model=None)
        sample_end = time.time()

        # log picked samples
        log_picked_samples(dest_dir_name, sample_rows, episode_id)



        sample_time = sample_end - sample_start

        # update the labeled pool
        labeled_rows = np.concatenate((labeled_rows,sample_rows),axis=0)
        np.savetxt(labeled_csv, labeled_rows,'%s,%s',delimiter=',', encoding='utf-8')


        # update the unlabeled pool
        unlabeled_rows = remove_rows(unlabeled_rows, sample_rows)
        np.savetxt(unlabeled_csv, unlabeled_rows, '%s,%s', delimiter=',', encoding='utf-8')

        print("Unlabeled pool size: ",len(unlabeled_rows))
        print("Labeled pool size: ",len(labeled_rows))


        #train the model
        dataset_train = MNIST(args.dataset_root, subset='train',csv_file='labeled.csv',transform=data_transforms)
        train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, **kwargs)
  

        model = Net().to(device) # initialize the model.
        optimizer = optim.Adam(model1.parameters(), lr=args.lr) # setup the optimizer
        # scheduler = StepLR(optimizer, step_size = 1, gamma=args.gamma)

        for epoch in range(1, args.epochs + 1):
            model1 = train(args, model1, device, train_loader, optimizer, epoch)
        accuracy = test(args, model1, device, test_loader)
            # scheduler.step()

        # save model
        save_path = os.path.join(dest_dir_name, 'ep_'+str(episode_id)+'.pth')
        torch.save(model.state_dict(), save_path)

        log(dest_dir_name, episode_id, args.sampling_method, sample_time, accuracy, labeled_rows)

        episode_id += 1
"""