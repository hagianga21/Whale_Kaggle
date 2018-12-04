from __future__ import print_function 
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from PIL import Image
from torch.utils.data import Dataset
from torch.autograd import Variable
from torchvision import transforms
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
import random
import torch.backends.cudnn as cudnn
from time import time
import cv2

class WhaleDataset(Dataset):
    def __init__(self, datafolder, filenames=None, y=None, transform=None):
        assert len(filenames) == len(y), "Number of files != number of labels"
        self.datafolder = datafolder
        self.y = y
        self.filenames = filenames
        self.image_files_list = [s for s in os.listdir(datafolder)]
        self.transform = transform

    def __len__(self):
        return len(self.image_files_list)

    def __getitem__(self, idx):
        print(idx)
        img_name = os.path.join(self.datafolder, self.filenames.values[idx][:])
        label = self.y[idx]

        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)
        print("ok")
        #image = cv2.imread(img_name)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = self.transform(image=image)['image']
        return image, self.y[idx]

def prepare_labels(y):
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # print(integer_encoded)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # print(onehot_encoded)

    y = onehot_encoded
    # print(y.shape)
    return y, label_encoder

class MyResNet(nn.Module):
    def __init__(self, depth, num_classes, pretrained = True):
        super(MyResNet, self).__init__()
        if depth == 18:
            model = models.resnet18(pretrained)
        elif depth == 34:
            model = models.resnet34(pretrained)
        elif depth == 50:
            model = models.resnet50(pretrained)
        elif depth == 152:
            model = models.resnet152(pretrained)

        self.num_ftrs = model.fc.in_features
        # self.num_classes = num_classes

        self.shared = nn.Sequential(*list(model.children())[:-1])
        self.target = nn.Sequential(nn.Linear(self.num_ftrs, num_classes))

    def forward(self, x):
        # pdb.set_trace()

        x = self.shared(x)
        x = torch.squeeze(x)
        return self.target(x)

    def frozen_until(self, to_layer):
        print('Frozen shared part until %d-th layer, inclusive'%to_layer)

        # if to_layer = -1, frozen all
        child_counter = 0
        for child in self.shared.children():
            if child_counter <= to_layer:
                print("child ", child_counter, " was frozen")
                for param in child.parameters():
                    param.requires_grad = False
                # frozen deeper children? check
                # https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/
            else:
                print("child ", child_counter, " was not frozen")
                for param in child.parameters():
                    param.requires_grad = True
            child_counter += 1

def unparallelize_model(model):
    try:
        while 1:
            # to avoid nested dataparallel problem
            model = model.module
    except AttributeError:
        pass
    return model

def net_frozen(args, model):
    print('********************************************************')
    model.frozen_until(args.frozen_until)
    init_lr = args.lr
    if args.trainer.lower() == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                lr=init_lr, weight_decay=args.weight_decay)
    elif args.trainer.lower() == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                lr=init_lr,  weight_decay=args.weight_decay)
    print('********************************************************')
    return model, optimizer

def parallelize_model(model):
    if torch.cuda.is_available():
        model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
    return model

def mytopk(pred, gt, k=3):
    """
    compute topk error
    pred: (n_sample,n_class) np array
    gt: a list of ground truth
    --------
    return:
        n_correct: number of correct prediction 
        topk_error: error, = n_connect/len(gt)
    """
    # topk = np.argpartition(pred, -k)[:, -k:]
    topk = np.argsort(pred, axis = 1)[:, -k:][:, ::-1]
    diff = topk - np.array(gt).reshape((-1, 1))
    n_correct = np.where(diff == 0)[0].size 
    topk_error = float(n_correct)/pred.shape[0]
    return n_correct, topk_error

def second2str(second):
    h = int(second/3600.)
    second -= h*3600.
    m = int(second/60.)
    s = int(second - m*60)
    return "{:d}:{:02d}:{:02d} (s)".format(h, m, s)

def print_eta(t0, cur_iter, total_iter):
    """
    print estimated remaining time
    t0: beginning time
    cur_iter: current iteration
    total_iter: total iterations
    """
    time_so_far = time() - t0
    iter_done = cur_iter + 1
    iter_left = total_iter - cur_iter - 1
    second_left = time_so_far/float(iter_done) * iter_left
    s0 = 'Epoch: '+ str(cur_iter + 1) + '/' + str(total_iter) + ', time so far: ' \
        + second2str(time_so_far) + ', estimated time left: ' + second2str(second_left)
    print(s0)

def cvt_to_gpu(X):
    return Variable(X.cuda()) if torch.cuda.is_available() \
        else Variable(X)

if __name__ == "__main__":
    input_size = 224 
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomSizedCrop(input_size),
            transforms.RandomHorizontalFlip(),  # simple data augmentation
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ]),
        'val': transforms.Compose([
            transforms.Scale(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ]),
    }

    train_df = pd.read_csv("../data/train.csv")
    print(train_df.values[1][0])
    y, label_encoder = prepare_labels(train_df['Id'])
    print('Split data...')
    train_img, val_img, train_labels, val_labels = train_test_split(train_df['Image'], train_df['Id'], test_size=0.2, random_state=2)
    #print("Size of Train set: ", train_img.values[24328][:])
    print("Size of Train set: ", train_img.head)
    print("ABC: ", train_img.values[8493][:])
    print("ABC: ", val_img[17])
    print("ABC: ", train_img[17])
    #print("Size of Valid set: ", train_labels.shape)