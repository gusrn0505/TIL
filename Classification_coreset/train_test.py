from __future__ import print_function, division
import torch

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np 
import torch.nn as nn

mixup_alpha = 4


class SC1_LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(SC1_LabelSmoothingCrossEntropy, self).__init__()
    def forward(self, y, targets, smoothing=0.1): # y는 hard labeling. SC2 도 hard labeling 형태로 반환해야겠네 
        confidence = 1. - smoothing
        log_probs = F.log_softmax(y, dim=-1) # 예측 확률 계산
        true_probs = torch.zeros_like(log_probs)
        true_probs.fill_(smoothing / (y.shape[1] - 1))
        true_probs.scatter_(1, targets.data.unsqueeze(1), confidence) # 정답 인덱스의 정답 확률을 confidence로 변경
        return torch.mean(torch.sum(true_probs * -log_probs, dim=-1)) # negative log likelihood


def mixup_data(x, y):
    lam = np.random.beta(mixup_alpha, mixup_alpha) # scalar 값 
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda() # shuffle 한 index 반환 
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def MNIST_train(args, model, device, labeled_dataset, labeled_dataset_label, optimizer, criterion, epoch):
    model.train()
    

    labeled_dataset = torch.tensor(labeled_dataset)
    labeled_dataset_label = torch.tensor(labeled_dataset_label)
    
    all_data = [(labeled_dataset[i], labeled_dataset_label[i][0]) for i in range(len(labeled_dataset_label))]
    
    if criterion == "hard labeling" : batch_size = 4
    elif criterion == "SC1" : batch_size = 32
    else : batch_size = 100

    data_loader = DataLoader(all_data, batch_size= batch_size)

    for i, (data, target) in enumerate(data_loader):
        data = data.view(-1, 1, 28,28)
        target = target.type(torch.LongTensor)
        data, target = data.to(device), target.to(device)
        data, target_a, target_b, lam = mixup_data(data, target)

        optimizer.zero_grad()
        output = model(data) #여기가 문제가 생기는 지점 

        # loss 함수 수정 필요. 
        if criterion == "hard labeling" : loss = mixup_criterion(F.nll_loss, output, target_a, target_b, lam)
        else : loss = mixup_criterion(SC1_LabelSmoothingCrossEntropy(), output, target_a, target_b, lam)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i, len(data_loader),
                100. * i / len(data_loader), loss.item()))
    return model

def MNIST_test(args, model, device, test_dataset, test_dataset_label, criterion) :
    model.eval()
    test_loss = 0
    correct = 0

    test_dataset = torch.tensor(test_dataset)
    test_dataset_label = torch.tensor(test_dataset_label)

    all_data = [(test_dataset[i], test_dataset_label[i][0]) for i in range(len(test_dataset_label))]
    data_loader = DataLoader(all_data, batch_size=100)

    # dataloader에 index가 가능한가? 
    with torch.no_grad():
        for data, target in data_loader:
            target = target.type(torch.LongTensor)
            data = data.view(-1, 1, 28,28)
            data, target = data.to(device), target.to(device)

            output = model(data)
            if criterion == "hard labeling" : F.nll_loss(output, target, reduction='sum').item()
            else : 
                criterion = SC1_LabelSmoothingCrossEntropy()
                test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_dataset),
        100. * correct / len(test_dataset)))

    return correct/len(test_dataset)



def CIFAR_train(args, model, device, labeled_dataset, labeled_dataset_label, optimizer, criterion, epoch):
    model.train()
    
    # for 절 자체에 오류가 나는 걸까, 아님 아래의 내용들에서 문제가 생기는 걸까. 
    # batch learning을 해주고 싶은데 흠. 
    labeled_dataset = torch.tensor(labeled_dataset)
    labeled_dataset_label = torch.tensor(labeled_dataset_label)
    
    all_data = [(labeled_dataset[i], labeled_dataset_label[i][0]) for i in range(len(labeled_dataset_label))]
    data_loader = DataLoader(all_data, batch_size=32)

    for i, (data, target) in enumerate(data_loader):
        data = data.view(len(target), 3, 32,32)
        target = target.type(torch.LongTensor)
        data, target = data.to(device), target.to(device)
        data, target_a, target_b, lam = mixup_data(data, target)

        optimizer.zero_grad()
        output = model(data) #여기가 문제가 생기는 지점 
        loss = mixup_criterion(criterion, output, target_a, target_b, lam)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i, len(data_loader),
                100. * i / len(data_loader), loss.item()))
    return model


def CIFAR_test(args, model, device, test_dataset, test_dataset_label, criterion) :
    model.eval()
    test_loss = 0
    correct = 0

    test_dataset = torch.tensor(test_dataset)
    test_dataset_label = torch.tensor(test_dataset_label)

    all_data = [(test_dataset[i], test_dataset_label[i][0]) for i in range(len(test_dataset_label))]
    data_loader = DataLoader(all_data, batch_size=100)

    # dataloader에 index가 가능한가? 
    with torch.no_grad():
        for data, target in data_loader:
            data = data.view(len(target), 3, 32,32)
            target = target.type(torch.LongTensor)
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_dataset),
        100. * correct / len(test_dataset)))

    return correct/len(test_dataset)

