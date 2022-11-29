from __future__ import print_function, division
import random

# Dataset에서 랜덤으로 init_size 만큼 추출하기 
def obtain_init_pool(init_size, dataset):
    '''
    Go to the dataset root. Get train.csv
    shuffle train.csv and get the first "init_size" samples.
    create three new csv files -> init_pool.csv, labeled.csv and unlabeled.csv
    '''
    init_pool_size = init_size

    original_dataset = dataset[:]
    random.shuffle(original_dataset)

    labeled_dataset = original_dataset[:init_pool_size]
    unlabeled_dataset = original_dataset[init_pool_size:]
 
    return labeled_dataset, unlabeled_dataset
