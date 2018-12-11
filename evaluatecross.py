import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow
from sklearn.model_selection import train_test_split
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
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch Digital Mammography Training')
parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
parser.add_argument('--net_type', default='resnet', type=str, help='model')
parser.add_argument('--depth', default=50, choices = [18, 34, 50, 152], type=int, help='depth of model')
parser.add_argument('--weight_decay', default=5e-6, type=float, help='weight decay')
parser.add_argument('--finetune', '-f', action='store_true', help='Fine tune pretrained model')
parser.add_argument('--trainer', default='adam', type = str, help = 'optimizer')
parser.add_argument('--model_path', type=str, default = '1205_0030')
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_workers', default=8, type=int)
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

input_size = 224 
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
train_df = pd.read_csv("../data/train.csv")
y, label_encoder = prepare_labels(train_df['Id'])

test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        #transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


test_set = WhaleDataset(
    datafolder='../data/test/', 
    datatype='test', 
    transform=test_transforms
)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

old_model = './checkpoint/' + 'resnetcross' + '-%s' % (args.depth) + '_' + args.model_path + '.t7'
if os.path.isfile(old_model):
    print("| Load pretrained at  %s..." % old_model)
    checkpoint = torch.load(old_model, map_location=lambda storage, loc: storage)
    tmp = checkpoint['model']
    model = unparallelize_model(tmp)
    best_top3 = checkpoint['top3']
    print('previous top3\t%.4f'% best_top3)
    print('=============================================')

sub = pd.read_csv('../data/sample_submission.csv')
model = parallelize_model(model)
model.eval()
for (inputs, labels, name) in tqdm(test_loader):
    inputs = cvt_to_gpu(inputs)
    output = model(inputs)
    output = output.cpu().detach().numpy()
    for i, (e, n) in enumerate(list(zip(output, name))):
        sub.loc[sub['Image'] == n, 'Id'] = ' '.join(label_encoder.inverse_transform(e.argsort()[-5:][::-1]))
print(output.shape)
sub.to_csv('submission.csv', index=False)
print("Done")

'''
for (data, target, name) in tqdm(test_loader):
    data = cvt_to_gpu(data)
    output = model(data)
    output = output.cpu().detach().numpy()
    for i, (e, n) in enumerate(list(zip(output, name))):
        sub.loc[sub['Image'] == n, 'Id'] = ' '.join(lab_encoder.inverse_transform(e.argsort()[-5:][::-1]))
        
sub.to_csv('submission.csv', index=False)
'''
