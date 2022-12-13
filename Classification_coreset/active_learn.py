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
from network_architectures import MNIST_BN_32_64_256, RGB_48_96_192_gp, RGB_128_256_down_gp
from train_test import MNIST_train, CIFAR_train, MNIST_test, CIFAR_test
from coreset import Coreset_Greedy




def argparser():
    parser = argparse.ArgumentParser(description='Active Learning - Image Classification')
    # 파싱할 인자들 추가하기

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--num-contact-subgraph', type=int, default=0)

# batch size 수정 필요 
    parser.add_argument('--al-sampling-size', default=100, type=int,
                        help='number of samples to add in each iteration')

    parser.add_argument('--SC1', default=True, type=bool)
    parser.add_argument('--SC2', default=False, type=bool)
    parser.add_argument('--num-class', default=10, type=int)

    parser.add_argument('--max-density', default=0, type=float)
    parser.add_argument('--min-density', default=0, type=float)    


    parser.add_argument('--threshold', type=float, default=0.9)


    parser.add_argument('--dataset-name', default='CIFAR10', type=str,
                        help='dataset name')

    parser.add_argument('--output-dir', default='output', type=str,
                        help='dataset name')

# MNIST 용으로 하기 때문에 clas는 10개로 
    parser.add_argument('--nclasses', type=int, default=10, metavar='N',
                        help='number of classes in the dataset')


    parser.add_argument('--max-eps', type=int, default=1, metavar='N',
                        help='max episodes of active learning')


    parser.add_argument('--dropout-iterations', type=int, default=5, metavar='N',
                        help='dropout iterations for bald method')

    parser.add_argument('--dim-reduction', type=str, default="CAE", metavar='LR',
                        help='method of dimension reduction')



    parser.add_argument('--first-epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')

    parser.add_argument('--second-epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--third-epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')


    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')

    return parser

def get_features(model, dataset, device="cuda"):
    features = []

    # 모델 evaluation 과정에서 사용하지 않아야 하는 layer들을 알아서 off 시키는 함수 
    # model을 evaluate를 할 수 있도록 세팅시키는 것 
    model.eval()
    # torch.no_grad : autograd engine을 비활성화시켜 필요한 메모리를 줄여주고 연산속도를 증가시킴 
    
    dataloader = DataLoader(dataset, batch_size = 64) # 여기서 Dataloader로 보내는 구나. 

    with torch.no_grad() : 
        for sample in dataloader:
            sample = sample.clone().detach().to(device)
        
            output = model.get_codes(sample)
            features = features + list(output.cpu().numpy()) 
    
    return features


# uncertainty, margin, coreset 등 다양한 방식으로 sampling 하기 
def active_sample(unlabeled_dataset, labeled_dataset, sc1_labeled_dataset, sample_size, model=None, device="cuda"):
    labeled_features = get_features(model, labeled_dataset, device) # (img_name, features)
    unlabeled_features = get_features(model, unlabeled_dataset, device)# (img_name, features)
    sc1_labeled_features = get_features(model, sc1_labeled_dataset, device)# (img_name, features)


    all_features = labeled_features +  unlabeled_features + sc1_labeled_features
        # label data의 index가 어디까지인지 표기. 
    labeled_indices = np.arange(0,len(labeled_features))

    coreset = Coreset_Greedy(all_features, len(labeled_features))

        # unlabeled 데이터에서 sample_size 만큼 center point 뽑기, 당시 반지름 뽑기
    new_batch, max_distance = coreset.sample(labeled_indices, sample_size)

    sc1_labeled_sample_index = [] 
    unlabeled_sample_index = [] 

    for index in new_batch : 
        if index >= len(labeled_features) + len(unlabeled_features) : 
            sc1_labeled_sample_index.append(index - len(labeled_features) - len(unlabeled_features))
        else : unlabeled_sample_index.append(index - len(labeled_features))

    unlabeled_sample_index.sort() 
    sc1_labeled_sample_index.sort() 

    return unlabeled_sample_index, sc1_labeled_sample_index, max_distance


def make_subgraph(sampling_label, original_dataset, radii, model):
    x = [original_dataset[i[1]] for i in sampling_label] 
    dataset = original_dataset

    if model is not None : 
        x = get_features(model, x, device="cuda")
        dataset = get_features(model, dataset, device="cuda")

    dist = pairwise_distances(x,dataset, metric='euclidean')

    subgraph= dist.copy()
    density_subgraph = []
    for i, row in enumerate(dist) : 
        for j, distance in enumerate(row) : 
            if distance > radii or j == sampling_label[i][1] : subgraph[i,j] =int(0) 
            else : subgraph[i,j] = int(1) 
        
        density_subgraph.append(sum(subgraph[i]))
    

    return np.array(subgraph), density_subgraph




def adjacency_subgraph(sample_data, sample_label, radii, model, M) :  
    dataset = sample_data
    num_subgraph = len(sample_label)
    if model is not None : 
        dataset = get_features(model,dataset, device="cuda")
    
    dist = pairwise_distances(dataset, dataset, metric='euclidean')
    adj_dist = dist.copy()
    
    for i, row in enumerate(dist) : 
        for j, distance in enumerate(row) : 
            if distance >= 2*radii[0] : adj_dist[i,j] = int(0)   
            elif 2*radii[0] > distance and distance >= radii[0]  : 
                adj_dist[i,j] = int(1)
            elif i==j : adj_dist[i,j] = int(0) # 자기자신은 제거
            else : 
                print('Break')

    classified_subgraph_index = []


    for i in range(num_subgraph) : 
        i_sub_class = "x"
        adj_index = np.where(adj_dist[ :,i] ==1)[0] 
        if len(adj_index)==0 : continue 
        i_sub_class = sample_label[i][0]

        for j in adj_index :  
            if i_sub_class != sample_label[j][0] : 
                i_sub_class = "x"
                continue
        if i_sub_class != "x" and len(adj_index) >= M : 
            classified_subgraph_index.append(i)
    
    classified_label = [sample_label[i] for i in classified_subgraph_index]

    return dist, adj_dist, classified_subgraph_index, classified_label

def cal_prob(unlabeled_index, count_subgraph, device = "cuda") : 
    i_count_subgraph = count_subgraph[unlabeled_index].copy()
    num_iteration = len(i_count_subgraph)
    min_radius = i_count_subgraph[num_iteration-1][1]

    for i , p_count in enumerate(i_count_subgraph) : 
        i_count_subgraph[i] = p_count[0] / ((i+1) *p_count[1] / (min_radius *num_iteration))
        i_count_subgraph[i] = F.softmax(torch.Tensor(i_count_subgraph[i]).to(device), dim=0)
        i_count_subgraph[i] = i_count_subgraph[i].cpu()
    
    if num_iteration == 1 : result = torch.tensor(i_count_subgraph[0])
        
    else : result = np.sum(i_count_subgraph, axis=0) / len(i_count_subgraph)
    return result 


def first_classification(classified_subgraph_index, pseudo_class_label, subgraph, density_subgraph, min_density, max_density) : 
    classification = defaultdict(list)
    dense_classified_subgraph = [density_subgraph[i] for i in classified_subgraph_index]    
    
    # 밀도가 높은 것이 앞으로 놓여져 있다. 
    sort_by_density = sorted(dense_classified_subgraph, reverse=True)

    # sort_by_density가 없는 경우 제외 
    if len(sort_by_density) ==0 : return classification
    lower_rank = int((1-min_density)*len(sort_by_density)) 
    upper_rank = int((max_density)*len(sort_by_density))

    upper_bound = sort_by_density[max(upper_rank-1, 0)]
    lower_bound = sort_by_density[max(lower_rank-1, 0)] # 밀도 상위 M % 의 subgraph만을 사용. 

    for i, index in enumerate(classified_subgraph_index) : 
        if density_subgraph[index] < lower_bound or density_subgraph[index] > upper_bound : continue
        x_index = list(np.where(subgraph[index] == 1)[0])
        label = pseudo_class_label[i][0]
    
        classification[label] += x_index
    
    # 중복 제거 및 정렬 
    for i in classification.keys() : 
        classification[i] = sorted(list(set(classification[i])))

    return classification

def second_classification(unlabeled_dataset_label, count_subgraph, threshold) : 
    unlabeled_index = [i[1] for i in unlabeled_dataset_label]
    classsification = defaultdict(list)

    for index in unlabeled_index : 
        prob = cal_prob(index, count_subgraph)
        if torch.max(prob) > threshold : 
            classsification[torch.argmax(prob)].append(index) 
    
    return classsification


def check_performance(classification, original_label) : 
    if len(classification) ==0 : return 0, 0, 0
    
    score = defaultdict(list) 
    all_score = 0 
    all_count = 0 
    for i in sorted(list(classification.keys())) : 
        x_index = classification[i] 
        num_x = len(x_index)
        count = 0 
        for index in x_index :
            if original_label[index][0] == i : count += 1 
        if num_x == 0 : i_score = 0
        else : i_score = count/num_x
        all_score += count
        all_count += num_x
        score[i] = [num_x, i_score]
    
    if all_count == 0 : return 0,0,0
    else : all_score = all_score/all_count
    
    return all_count, all_score, score

def update_count_subgraph(count_subgraph, unlabeled_dataset_label, labeled_dataset_label, subgraph, radius) : 
    unlabeled_index = [i[1] for i in unlabeled_dataset_label]
    for i in unlabeled_index : 
        count = [0]*10
        i_subgraph = np.where(subgraph[:, i]==1)[0]
        
        for j in i_subgraph : 
            count[labeled_dataset_label[j][0]] += 1
        if sum(count) != 0 : count_subgraph[i].append([count, radius[0]]) 

    return count_subgraph



def mixup_data(x, y, mixup_alpha =4):
    lam = np.random.beta(mixup_alpha, mixup_alpha) # scalar 값 
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda() # shuffle 한 index 반환 
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)




# 수정 필요! 
class SC2_LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(SC2_LabelSmoothingCrossEntropy, self).__init__()
        
    def forward(self, prob, _): # y는 hard labeling. SC2 도 hard labeling 형태로 반환해야겠네 
        prob = torch.tensor(prob)
        log_probs = F.log_softmax(prob, dim=-1) # 예측 확률 계산
        return torch.mean(torch.sum(prob * -log_probs, dim=-1)) # negative log likelihood

# dest_dir 위치에 결과 기록하기. 입력할 결과들을 입력값으로 넣음 
def log(dest_dir, episode_id, sample_method,  label_dataset_label, num_classification, max_density, min_density, accuracy):
    # log file은 dest_dir 위치에 log.csv를 두기 위한 주소이다. 
    log_file = os.path.join(dest_dir, 'log.csv')

    # 주소가 정확하지 않을  해당 위치에 파일이 존재하지 않을 때, log_rows를 다음과 같이 정한다. 
    if not os.path.exists(log_file):
        log_rows = [['Episode Id','Sample Method','Labeled Pool', 'Num of classification', 'max_density', 'min_density', 'Accuracy']]
    # 파일이 존재할 때에는 데이터를 처리해서 불러온다. 
    else:
        log_rows = np.genfromtxt(log_file, delimiter=',', dtype=str, encoding='utf-8').tolist()

    # episod_id, sample_mthod, sample_time 등의 값을 추가한다. 
    log_rows.append([episode_id,sample_method, len(label_dataset_label), num_classification, max_density, min_density, accuracy])
    
    # 데이터를 저장한다. 파일이 없다면 새로 만든다. 
    np.savetxt(log_file,log_rows,'%s,%s,%s,%s,%s,%s,%s',delimiter=',')


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



    # 데이터 셋 변경 시 수정 필요 ####################################
    original_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.MNIST( 
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    # MNIST Dataset을 가공할 수 있는 list로 변경. feature와 label 각각 저장 
    original_all = []
    original_dataset = []
    original_label = [] 

    test_dataset = [] 
    test_label = [] 


    for i, sample in enumerate(original_data) : 
        original_all.append(sample)
        feature = np.array(sample[0])
        original_dataset.append(feature)
        original_label.append([sample[1], i]) # 여기다가 index를 추가해버릴까?  (label, index) 이런 형태로! 
    
    for i, test in enumerate(test_data) : 
        feature = np.array(test[0])
        test_dataset.append(feature)
        test_label.append([test[1], i])

    unlabeled_dataset = original_dataset[:]
    unlabeled_dataset_label = original_label[:]

    labeled_dataset = [] 
    labeled_dataset_label = []

    sc1_labeled_dataset = [] 
    sc1_labeled_dataset_label = []

    sc2_labeled_dataset = [] 
    sc2_labeled_dataset_label = []


    count_subgraph = defaultdict(list)

    # 데이터 셋 변경 시 수정 필요 #####################################
    PATH = './weights/MNIST/'
    AE = None
    CAE = torch.load(PATH + 'CAE.pt')  
    CAE.load_state_dict(torch.load(PATH + 'CAE_state_dict.pt'))  

    print("Successfully loaded CAE")

    episode_id =1 
    # set the sample size
    # Iteration 과 같이 고려하여 총 Label data의 개수로 반환할 수 있도록 할 것 
    sample_size = args.al_sampling_size
    if len(unlabeled_dataset) < sample_size:
        sample_size = len(unlabeled_dataset)
    
    # 첫 Coreset selection 
    un_sample_index, c_sample_index, radius  = active_sample(unlabeled_dataset, labeled_dataset, sc1_labeled_dataset, sample_size, model=CAE, device=device)

    sample_data = [unlabeled_dataset[i] for i in un_sample_index]
    sample_label = [unlabeled_dataset_label[i] for i in un_sample_index]

    for i in un_sample_index[::-1] : 
        del unlabeled_dataset[i]
        del unlabeled_dataset_label[i]

    if len(c_sample_index) != 0 : 
        c_sample_data = [sc1_labeled_dataset[i] for i in c_sample_index]
        c_sample_label = [sc1_labeled_dataset_label[i] for i in c_sample_index]
        sample_data = np.concatenate((sample_data, c_sample_data), axis=0) 
        sample_label = np.concatenate((sample_label, c_sample_label), axis=0) 

        for i in c_sample_index[::-1] : 
            np.delete(sc1_labeled_dataset, i, axis=0)
            np.delete(sc1_labeled_dataset_label, i, axis=0)

    if len(labeled_dataset_label) == 0 :  
        labeled_dataset = sample_data[:]
        labeled_dataset_label = sample_label[:]
    else : 
        labeled_dataset = np.concatenate((labeled_dataset,sample_data),axis=0)
        labeled_dataset_label = np.concatenate((labeled_dataset_label, sample_label), axis =0)

    # subgraph 형성 및 adjacency 확인 
    subgraph, density_subgraph = make_subgraph(labeled_dataset_label, original_dataset, radius, CAE)
    dist_class, adj_dist, classified_subgraph_index, pseudo_class_label = adjacency_subgraph(labeled_dataset, labeled_dataset_label, radius, CAE, 0)

    print("First coreset selection work!")


    # 밀도 측정. 실험 이후 조정 필요 
    #ratio_term = list(np.arange(0.01, 1-args.max_density, 0.01))
    # 다음은 max_density를 조정해보자. 
    #for ratio in ratio_term : 
    #    if len(pseudo_class_label) == 0 : break 
    #    f_classification = first_classification(classified_subgraph_index, pseudo_class_label, subgraph, density_subgraph, ratio, args.max_density)
    #    sc1_num_classification, sc1_score, sc1_dic_score = check_performance(f_classification,original_label)

    #    log(dest_dir_name, episode_id, "CS1", labeled_dataset_label, sc1_num_classification, args.max_density, ratio, sc1_score)

    f_classification = first_classification(classified_subgraph_index, pseudo_class_label, subgraph, density_subgraph, args.min_density, args.max_density)
    sc1_num_classification, sc1_score, sc1_dic_score = check_performance(f_classification,original_label)
    log(dest_dir_name, episode_id, "CS1", labeled_dataset_label, sc1_num_classification, args.max_density, args.min_density, sc1_score)


    # SC1 방법 이후 SC1 labeled 데이터셋 구분하기 
    if args.SC1 == True : 
        erase_dataset_ori_index = []
        for i in f_classification.keys(): 
            index = f_classification[i]

            new_labeled_dataset = [original_dataset[j] for j in index]
            new_labeled_dataset_label = [ [i,j] for j in index ]
            new_erase_original_index = [new_labeled_dataset_label[j][1] for j in range(len(new_labeled_dataset_label))]

            if len(new_labeled_dataset_label) ==0 : continue

            if len(sc1_labeled_dataset_label) == 0 : 
                sc1_labeled_dataset = new_labeled_dataset
                sc1_labeled_dataset_label = new_labeled_dataset_label
        
            else : 
                sc1_labeled_dataset = np.concatenate((sc1_labeled_dataset, new_labeled_dataset), axis=0)
                sc1_labeled_dataset_label = np.concatenate((sc1_labeled_dataset_label, new_labeled_dataset_label), axis =0)
    
            erase_dataset_ori_index += new_erase_original_index

        erase_unlabeled_index = [np.where(np.array(unlabeled_dataset_label).T[1] == i)[0][0]  for i in erase_dataset_ori_index]
        erase_unlabeled_index.sort()

        for i in erase_unlabeled_index[::-1] : 
            del unlabeled_dataset[i]
            del unlabeled_dataset_label[i]

    # 각 unlabeled 데이터에 대해서 각 Class / radius 기록하기 
    update_count_subgraph(count_subgraph, original_label, labeled_dataset_label, subgraph, radius)

    f_score_lst = [] 
    f_score_lst.append([sc1_num_classification, sc1_score])


    # Iteration 시작 
    episode_id = 2
    while True:
        if episode_id > args.max_eps:
            break

        print("Episode #",episode_id)

        # sanity checks
        if len(unlabeled_dataset_label) == 0: break

        # set the sample size
        # Sampling 방식에 따라 변화필요함. 
        sample_size = args.al_sampling_size
        if len(unlabeled_dataset) < sample_size:
            sample_size = len(unlabeled_dataset_label)

        # active learning 간 label data 외에 전 데이터에 대해서 적용하기 

        #except_labeled_dataset = np.concatenate((unlabeled_dataset, sc1_labeled_dataset), axis=0)

        # active sampling 및 그에 따른 dataset 변경
        un_sample_index, c_sample_index, radius  = active_sample(unlabeled_dataset, labeled_dataset, sc1_labeled_dataset, sample_size, model=CAE, device=device)

        sample_data = [unlabeled_dataset[i] for i in un_sample_index]
        sample_label = [unlabeled_dataset_label[i] for i in un_sample_index]

        for i in un_sample_index[::-1] : 
            del unlabeled_dataset[i]
            del unlabeled_dataset_label[i]

        if len(c_sample_index) != 0 : 
            c_sample_data = [sc1_labeled_dataset[i] for i in c_sample_index]
            c_sample_label = [sc1_labeled_dataset_label[i] for i in c_sample_index]
            sample_data = np.concatenate((sample_data, c_sample_data), axis=0) 
            sample_label = np.concatenate((sample_label, c_sample_label), axis=0) 

            for i in c_sample_index[::-1] : 
                np.delete(sc1_labeled_dataset, i, axis=0)
                np.delete(sc1_labeled_dataset_label, i, axis=0)

        if len(labeled_dataset_label) == 0 :  
            labeled_dataset = sample_data[:]
            labeled_dataset_label = sample_label[:]
        else : 
            labeled_dataset = np.concatenate((labeled_dataset,sample_data),axis=0)
            labeled_dataset_label = np.concatenate((labeled_dataset_label, sample_label), axis =0)


        # 
        subgraph, density_subgraph = make_subgraph(labeled_dataset_label, original_dataset, radius, CAE)
        dist_class, adj_dist, classified_subgraph_index, pseudo_class_label = adjacency_subgraph(labeled_dataset, labeled_dataset_label, radius, CAE, 0)

        
        ratio_term = list(np.arange(0.01, 1-args.max_density, 0.01))
        for ratio in ratio_term : 
            if len(pseudo_class_label) == 0 : break 
            f_classification = first_classification(classified_subgraph_index, pseudo_class_label, subgraph, density_subgraph, ratio, args.max_density)
            sc1_num_classification, sc1_score, sc1_dic_score = check_performance(f_classification,original_label)

            log(dest_dir_name, episode_id, "CS1", labeled_dataset_label, sc1_num_classification, args.max_density, ratio, sc1_score)

        f_score_lst.append([sc1_num_classification, sc1_score])


        f_classification = first_classification(classified_subgraph_index, pseudo_class_label, subgraph, density_subgraph, args.min_density, args.max_density)
        sc1_num_classification, sc1_score, sc1_dic_score = check_performance(f_classification,original_label)
        log(dest_dir_name, episode_id, "CS1", labeled_dataset_label, sc1_num_classification, args.max_density, args.min_density, sc1_score)

        # SC1 방법에 따른 sc1_labeled data 제거 
        if args.SC1 == True : 
            erase_dataset_ori_index = []
            pre_index = [j[1] for j in sc1_labeled_dataset_label]

            for i in f_classification.keys(): 
                index = f_classification[i]
                index = list(set(index) - set(pre_index))

                new_labeled_dataset = [original_dataset[j] for j in index]
                new_labeled_dataset_label = [ [i,j] for j in index ]
                new_erase_original_index = [new_labeled_dataset_label[j][1] for j in range(len(new_labeled_dataset_label))]

                if len(sc1_labeled_dataset_label) == 0 : 
                    sc1_labeled_dataset = new_labeled_dataset
                    sc1_labeled_dataset_label = new_labeled_dataset_label
        
                elif len(new_labeled_dataset_label) ==0 : 
                    sc1_labeled_dataset = sc1_labeled_dataset
                    sc1_labeled_dataset_label = sc1_labeled_dataset_label               

                else : 
                    sc1_labeled_dataset = np.concatenate((sc1_labeled_dataset, new_labeled_dataset), axis=0)
                    sc1_labeled_dataset_label = np.concatenate((sc1_labeled_dataset_label, new_labeled_dataset_label), axis =0)
    
                erase_dataset_ori_index += new_erase_original_index

            erase_unlabeled_index = [np.where(np.array(unlabeled_dataset_label).T[1] == i)[0][0]  for i in erase_dataset_ori_index]
            erase_unlabeled_index.sort()

            for i in erase_unlabeled_index[::-1] : 
                del unlabeled_dataset[i]
                del unlabeled_dataset_label[i]

        update_count_subgraph(count_subgraph, original_label, labeled_dataset_label, subgraph, radius)

        episode_id += 1


    # sc2 방법 적용 
    #threshold_term = list(np.arange(0.01, 1, 0.01))
    #for threshold in threshold_term : 
    #    sc2_classification = second_classification(unlabeled_dataset_label, count_subgraph, threshold)
    #    sc2_num_classification, sc2_score, sc2_dic_score = check_performance(sc2_classification, original_label)
    #    log(dest_dir_name, episode_id, "CS2", labeled_dataset_label, sc2_num_classification, threshold, 0, sc2_score)
   
    sc2_classification = second_classification(unlabeled_dataset_label, count_subgraph, args.threshold)
    sc2_num_classification, sc2_score, sc2_dic_score = check_performance(sc2_classification, original_label)
 
    print("SC2 performance : num_classification", sc2_num_classification, "score", sc2_score )
    
    if args.SC2 == True : 
        erase_dataset_ori_index = []
        for i in sc2_classification.keys(): 
            index = sc2_classification[i]

            new_labeled_dataset = [original_dataset[j] for j in index]
            new_labeled_dataset_label = [ [i,j] for j in index ]

            if len(sc2_labeled_dataset_label) == 0 : 
                sc2_labeled_dataset = new_labeled_dataset[:]
                sc2_labeled_dataset_label = new_labeled_dataset_label[:]

            else : 
                sc2_labeled_dataset = np.concatenate((sc2_labeled_dataset, new_labeled_dataset), axis=0)
                sc2_labeled_dataset_label = np.concatenate((sc2_labeled_dataset_label, new_labeled_dataset_label), axis=0)

            erase_dataset_ori_index += index

        erase_unlabeled_index = [np.where(np.array(unlabeled_dataset_label).T[1] == i)[0][0]  for i in erase_dataset_ori_index]
        erase_unlabeled_index.sort()

        for i in erase_unlabeled_index[::-1] : 
            del unlabeled_dataset[i]
            del unlabeled_dataset_label[i]

    log(dest_dir_name, episode_id, "CS2", labeled_dataset_label, sc2_num_classification, args.max_density, args.min_density, sc2_score)

    total = 0
    total_score = 0 
    for (num, score) in f_score_lst : 
        total += num
        total_score += num*score
    final_sc1_score = total_score / total
    
    print("label :", len(labeled_dataset_label), "SC1 :", len(sc1_labeled_dataset_label), final_sc1_score, "SC2 :", sc2_num_classification, sc2_score)


    criterion = "hard labeling"
    print("Only labeled data")

    # 여기서 부턴 CNN 모델 학습!  
    neural_1 = MNIST_BN_32_64_256(10).to(device)
    #neural_1 = RGB_48_96_192_gp(10).to(device)
    #neural = RGB_128_256_down_gp.to(device)

    optimizer1 = optim.Adam(neural_1.parameters(), lr=args.lr) # setup the optimizer
    scheduler1 = StepLR(optimizer1, step_size = 10, gamma=args.gamma)

    # Label data만 사용  
    for epoch in range(1, args.first_epochs + 1):
        neural_1 = MNIST_train(args, neural_1, device, labeled_dataset, labeled_dataset_label, optimizer1, criterion, epoch)        
        #neural_1 = CIFAR_train(args, neural_1, device, labeled_dataset, labeled_dataset_label, optimizer1, criterion, epoch)        
        
        scheduler1.step()

    accuracy = MNIST_test(args, neural_1, device, test_dataset, test_label, criterion)
    #accuracy = CIFAR_test(args, neural_1, device, test_dataset, test_label, criterion)
    log(dest_dir_name, episode_id, "CNN", labeled_dataset_label, len(unlabeled_dataset_label), 0,0, accuracy)


    # label data + SC1 

    optimizer2 = optim.Adam(neural_1.parameters(), lr=args.lr ) # setup the optimizer
    scheduler2 = StepLR(optimizer1, step_size = 10, gamma=args.gamma)
    
    if args.SC1 == True : 
        print("labeled data + SC1 labeling data")
        criterion = "SC1" 
        for epoch in range(1, args.second_epochs + 1):
            neural_1 = MNIST_train(args, neural_1, device, sc1_labeled_dataset, sc1_labeled_dataset_label, optimizer1, criterion, epoch)        
            #neural_1 = CIFAR_train(args, neural_1, device, sc1_labeled_dataset, sc1_labeled_dataset_label, optimizer1, criterion, epoch)        
            scheduler1.step()

        accuracy = MNIST_test(args, neural_1, device, test_dataset, test_label, criterion)
        #accuracy = CIFAR_test(args, neural_1, device, test_dataset, test_label, criterion)

        log(dest_dir_name, episode_id, "CNN", sc1_labeled_dataset_label, len(unlabeled_dataset_label), 0, 0, accuracy)
    
    if args.SC2 == True : 
        print("labeled data + SC1 labeling data + SC2 labeling data")
        criterion = "SC2" 
        # Label data + SC1 + SC2 
        for epoch in range(1, args.third_epochs + 1):
            neural_1 = MNIST_train(args, neural_1, device, sc2_labeled_dataset, sc2_labeled_dataset_label, optimizer1, criterion, epoch)        
            #neural_1 = CIFAR_train(args, neural_1, device, sc2_labeled_dataset, sc2_labeled_dataset_label, optimizer1, criterion, epoch)        
            scheduler1.step()

        accuracy = MNIST_test(args, neural_1, device, test_dataset, test_label, criterion)
        #accuracy = CIFAR_test(args, neural_1, device, test_dataset, test_label, criterion)

        log(dest_dir_name, episode_id, "CNN", sc2_labeled_dataset_label, len(unlabeled_dataset_label), 0, 0, accuracy)



