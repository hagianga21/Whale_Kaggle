import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pandas as pd 
import torch
import torch.nn as nn
import argparse
import copy
import random
from torchvision import transforms
# import time
import torch.backends.cudnn as cudnn
import os, sys
from time import time, strftime
from data_utils import *


parser = argparse.ArgumentParser(description='PyTorch Digital Mammography Training')
parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
parser.add_argument('--net_type', default='resnet', type=str, help='model')
parser.add_argument('--depth', default=50, choices = [18, 34, 50, 152], type=int, help='depth of model')
parser.add_argument('--weight_decay', default=5e-6, type=float, help='weight decay')
parser.add_argument('--finetune', '-f', action='store_true', help='Fine tune pretrained model')
parser.add_argument('--trainer', default='adam', type = str, help = 'optimizer')
parser.add_argument('--model_path', type=str, default = ' ')
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--num_epochs', default=1500, type=int,
                    help='Number of epochs in training')
parser.add_argument('--dropout_keep_prob', default=0.5, type=float)
parser.add_argument('--check_after', default=2,
                    type=int, help='check the network after check_after epoch')
parser.add_argument('--train_from', default=1,
                    choices=[0, 1, 2],  # 0: from scratch, 1: from pretrained Resnet, 2: specific checkpoint in model_path
                    type=int,
                    help="training from beginning (1) or from the most recent ckpt (0)")
parser.add_argument('--frozen_until', '-fu', type=int, default = 8,
                    help="freeze until --frozen_util block")
parser.add_argument('--val_ratio', default=0.1, type=float, 
        help = "number of training samples per class")
args = parser.parse_args()

KTOP = 5 # top k error
maxEpoches = args.num_epochs
numSplit = 10


def exp_lr_scheduler(args, optimizer, epoch):
    # after epoch 100, not more learning rate decay
    init_lr = args.lr
    lr_decay_epoch = 4 # decay lr after each 10 epoch
    weight_decay = args.weight_decay
    lr = init_lr * (0.6 ** (min(epoch, 200) // lr_decay_epoch)) 

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['weight_decay'] = weight_decay

    return optimizer, lr

use_gpu = torch.cuda.is_available()
print('Loading data...')
train_df = pd.read_csv("../data/train.csv")
y, label_encoder = prepare_labels(train_df['Id'])
train_labels = y

print(f"There are {len(os.listdir('../data/train'))} images in train dataset with {train_df.Id.nunique()} unique classes.")
print(f"There are {len(os.listdir('../data/test'))} images in test dataset.")

print('Split data...')
kford = KFold(numSplit, shuffle = True, random_state=2)

X = train_df['Image'].values
print(X.shape)
print(y.shape)

for train, val in kford.split(X):
    print('train: %s, test: %s' % (train, val))
    X[train]
    X[val]