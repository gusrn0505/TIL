from collections import OrderedDict
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init as nninit
import torchvision.models as models, torchvision.models.resnet as resnet

_ARCH_REGISTRY = {}


def architecture(name, sample_shape):
    """
    Decorator to register an architecture;

    Use like so:

    >>> @architecture('my_architecture', (3, 32, 32))
    ... class MyNetwork(nn.Module):
    ...     def __init__(self, n_classes):
    ...         # Build network
    ...         pass
    """
    def decorate(fn):
        _ARCH_REGISTRY[name] = (fn, sample_shape)
        return fn
    return decorate


def get_net_and_shape_for_architecture(arch_name):
    """
    Get network building function and expected sample shape:

    For example:
    >>> net_class, shape = get_net_and_shape_for_architecture('my_architecture')

    >>> if shape != expected_shape:
    ...     raise Exception('Incorrect shape')
    """
    return _ARCH_REGISTRY[arch_name]



@architecture('mnist-bn-32-64-256', (1, 28, 28))
class MNIST_BN_32_64_256 (nn.Module):
    def __init__(self, n_classes):
        super(MNIST_BN_32_64_256, self).__init__()

        self.conv1_1 = nn.Conv2d(1, 32, (5, 5))
        self.conv1_1_bn = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2_1 = nn.Conv2d(32, 64, (3, 3))
        self.conv2_1_bn = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, (3, 3))
        self.conv2_2_bn = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2, 2))

        self.drop1 = nn.Dropout()
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1_1_bn(self.conv1_1(x)))) #(1,28,28) -> #(32, 12,12)
        x = F.relu(self.conv2_1_bn(self.conv2_1(x))) # (32,12,12) -> (64, 10, 10)
        x = self.pool2(F.relu(self.conv2_2_bn(self.conv2_2(x)))) #(64, 10, 10) -> (64,4,4)
        x = x.view(-1, 1024) # Flatten 
        x = self.drop1(x)
        x = F.relu(self.fc3(x)) #(1024 -> 256) 
        x = self.fc4(x) # 256 -> n_classes
        output = F.log_softmax(x, dim=1)
        return output


@architecture('rgb-48-96-192-gp', (3, 32, 32))
class RGB_48_96_192_gp (nn.Module):
    def __init__(self, n_classes):
        super(RGB_48_96_192_gp, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 48, (3, 3), padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(48)
        self.conv1_2 = nn.Conv2d(48, 48, (3, 3), padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(48)
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2_1 = nn.Conv2d(48, 96, (3, 3), padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(96)
        self.conv2_2 = nn.Conv2d(96, 96, (3, 3), padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(96)
        self.conv2_3 = nn.Conv2d(96, 96, (3, 3), padding=1)
        self.conv2_3_bn = nn.BatchNorm2d(96)
        self.pool2 = nn.MaxPool2d((2, 2))

        self.conv3_1 = nn.Conv2d(96, 192, (3, 3), padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(192)
        self.conv3_2 = nn.Conv2d(192, 192, (3, 3), padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(192)
        self.conv3_3 = nn.Conv2d(192, 192, (3, 3), padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(192)
        self.pool3 = nn.MaxPool2d((2, 2))

        self.drop1 = nn.Dropout()

        self.fc4 = nn.Linear(192, 192)
        self.fc5 = nn.Linear(192, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1_1_bn(self.conv1_1(x))) # (3,32,32) -> (48,32,32)
        x = self.pool1(F.relu(self.conv1_2_bn(self.conv1_2(x)))) # (48,32,32) => (48,16,16)

        x = F.relu(self.conv2_1_bn(self.conv2_1(x))) #(48,16,16) => (96,16,16)
        x = F.relu(self.conv2_2_bn(self.conv2_2(x))) # (96,16,16) => (96,16,16)
        x = self.pool2(F.relu(self.conv2_3_bn(self.conv2_3(x)))) # (96,16,16) => (96,8,8)

        x = F.relu(self.conv3_1_bn(self.conv3_1(x))) # (96,8,8) => (192, 8,8)
        x = F.relu(self.conv3_2_bn(self.conv3_2(x))) # (192, 8,8) => (192, 8,8)
        x = self.pool3(F.relu(self.conv3_3_bn(self.conv3_3(x)))) # (192, 8, 8) => (192, 4,4)

        x = F.avg_pool2d(x, 4) # (192, 4, 4) => (192, 1, 1)
        x = x.view(-1, 192)
        x = self.drop1(x)

        x = F.relu(self.fc4(x)) # (192, 192)
        x = self.fc5(x) # 192 -> n_classes
        output = F.log_softmax(x, dim=1)
        return output


@architecture('rgb-128-256-down-gp', (3, 32, 32))
class RGB_128_256_down_gp(nn.Module):
    def __init__(self, n_classes):
        super(RGB_128_256_down_gp, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 128, (3, 3), padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(128)
        self.conv1_2 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(128)
        self.conv1_3 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv1_3_bn = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.drop1 = nn.Dropout()

        self.conv2_1 = nn.Conv2d(128, 256, (3, 3), padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(256)
        self.conv2_2 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(256)
        self.conv2_3 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.conv2_3_bn = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d((2, 2))
        self.drop2 = nn.Dropout()

        self.conv3_1 = nn.Conv2d(256, 512, (3, 3), padding=0)
        self.conv3_1_bn = nn.BatchNorm2d(512)
        self.nin3_2 = nn.Conv2d(512, 256, (1, 1), padding=1)
        self.nin3_2_bn = nn.BatchNorm2d(256)
        self.nin3_3 = nn.Conv2d(256, 128, (1, 1), padding=1)
        self.nin3_3_bn = nn.BatchNorm2d(128)

        self.fc4 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1_1_bn(self.conv1_1(x)))
        x = F.relu(self.conv1_2_bn(self.conv1_2(x)))
        x = self.pool1(F.relu(self.conv1_3_bn(self.conv1_3(x)))) # (3,32,32) -> (128,16,16)
        x = self.drop1(x)

        x = F.relu(self.conv2_1_bn(self.conv2_1(x)))
        x = F.relu(self.conv2_2_bn(self.conv2_2(x)))
        x = self.pool2(F.relu(self.conv2_3_bn(self.conv2_3(x)))) # => (256,8,8)
        x = self.drop2(x)

        x = F.relu(self.conv3_1_bn(self.conv3_1(x))) # (256,8,8) => (512, 8,8)
        x = F.relu(self.nin3_2_bn(self.nin3_2(x))) # (512, 8,8) => (256, 10, 10) 
        x = F.relu(self.nin3_3_bn(self.nin3_3(x))) # (256, 10,10) => (128, 12, 12) 

        x = F.avg_pool2d(x, 6) 
        x = x.view(-1, 128)

        x = self.fc4(x)
        output = F.log_softmax(x, dim=1)
        return output



def robust_binary_crossentropy(pred, tgt):
    inv_tgt = -tgt + 1.0
    inv_pred = -pred + 1.0 + 1e-6
    return -(tgt * torch.log(pred + 1.0e-6) + inv_tgt * torch.log(inv_pred))

def bugged_cls_bal_bce(pred, tgt):
    inv_tgt = -tgt + 1.0
    inv_pred = pred + 1.0 + 1e-6
    return -(tgt * torch.log(pred + 1.0e-6) + inv_tgt * torch.log(inv_pred))

def log_cls_bal(pred, tgt):
    return -torch.log(pred + 1.0e-6)

def get_cls_bal_function(name):
    if name == 'bce':
        return robust_binary_crossentropy
    elif name == 'log':
        return log_cls_bal
    elif name == 'bug':
        return bugged_cls_bal_bce