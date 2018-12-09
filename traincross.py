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

KTOP = 5 # top k error
maxEpoches = args.num_epochs
numSplit = 10


def exp_lr_scheduler(args, optimizer, epoch):
    # after epoch 100, not more learning rate decay
    init_lr = args.lr
    lr_decay_epoch = 4 # decay lr after each 10 epoch
    weight_decay = args.weight_decay
    temp = epoch//10 #Sau 10 epoch moi thay learning rate 1 lan
    lr = init_lr * (0.6 ** (min(temp, 2000) // lr_decay_epoch)) 

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

img_train = dict()
label_train = dict()
img_val = dict()
label_val = dict()
k = 0

for train, val in kford.split(X):
    print('train: %s, test: %s' % (train, val))
    img_train[k] = X[train]
    img_val[k] = X[val]
    label_train[k] = y[train]
    label_val[k] = y[val]
    k = k + 1

print(img_train[1].shape)
print(img_train[1][1])

input_size = 224 
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

datatrain_transforms =  transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomRotation(20),
        transforms.RandomRotation((70, 90)),
        transforms.RandomAffine(20),
        transforms.RandomHorizontalFlip(),  # simple data augmentation
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.2),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])

dataval_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.2),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])

########## 
print('Load model')
saved_model_fn = 'resnetcross' + '-%s' % (args.depth) + '_' + strftime('%m%d') + '_' + '%s' %(args.batch_size)
old_model = './checkpoint/' + 'resnetcross' + '-%s' % (args.depth) + '_' + args.model_path + '.t7'
if args.train_from == 2 and os.path.isfile(old_model):
    print("| Load pretrained at  %s..." % old_model)
    checkpoint = torch.load(old_model, map_location=lambda storage, loc: storage)
    tmp = checkpoint['model']
    model = unparallelize_model(tmp)
    best_top3 = checkpoint['top3']
    print('previous top3\t%.4f'% best_top3)
    print('=============================================')
else:
    model = MyResNet(args.depth, 5005)

##################
print('Start training ... ')
criterion = nn.CrossEntropyLoss()
#criterion = nn.BCEWithLogitsLoss()
model, optimizer = net_frozen(args, model)
model = parallelize_model(model)

N_train = len(label_train[0])
N_valid = len(label_val[0])
best_top3 = 1 
best_epoch = 0
t0 = time()

ValidationTop1Error = np.ones(10)
ValidationTop5Error = np.ones(10)
for epoch in range(args.num_epochs):
    j = epoch%10
    print('Hello', j)
    train_dataset= WhaleDataset2(datafolder='../data/train/', datatype='train', filenames = img_train[j], labels = label_train[j], transform=datatrain_transforms)
    val_dataset= WhaleDataset2(datafolder='../data/train/', datatype='train', filenames = img_val[j], labels = label_val[j], transform=dataval_transforms)
    dset_loaders = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    valset_loaders = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)

    optimizer, lr = exp_lr_scheduler(args, optimizer, epoch) 
    print('#################################################################')
    print('=> Training Epoch #%d, LR=%.10f' % (epoch + 1, lr))

    running_loss, running_corrects, tot = 0.0, 0.0, 0.0
    runnning_topk_corrects = 0.0
    model.train()
    torch.set_grad_enabled(True)

    for batch_idx, (inputs, labels) in enumerate(dset_loaders):
        optimizer.zero_grad()
        inputs = cvt_to_gpu(inputs)
        labels = cvt_to_gpu(labels)
        outputs = model(inputs)
        #print(outputs.shape) 32 x 5005 
        #print(labels.shape) 32 x 5005
        #loss = criterion(outputs, labels.float())
        loss = criterion(outputs, torch.max(labels, 1)[1])
        running_loss += loss*inputs.shape[0]
        loss.backward()
        optimizer.step()
        ############################################
        _, preds = torch.max(outputs.data, 1)
        _, tmplabel = torch.max(labels.data, 1)

        # topk 
        top3correct, _ = mytopk(outputs.data.cpu().numpy(), tmplabel, KTOP)
        runnning_topk_corrects += top3correct
        # pdb.set_trace()
        running_loss += loss.item()
        running_corrects += preds.eq(tmplabel).cpu().sum()
        tot += labels.size(0)
        sys.stdout.write('\r')
        try:
            batch_loss = loss.item()
        except NameError:
            batch_loss = 0

        top1error = 1 - float(running_corrects)/tot
        top3error = 1 - float(runnning_topk_corrects)/tot

        sys.stdout.write('| Epoch [%2d/%2d] Iter [%3d/%3d]\tBatch loss %.4f\tTop1error %.4f \tTop3error %.4f\n'
                         % (epoch + 1, args.num_epochs, batch_idx + 1,
                            (len(os.listdir('../data/train')) // args.batch_size), batch_loss/args.batch_size,
                            top1error, top3error))
        sys.stdout.flush()
        sys.stdout.write('\r')

    top1error = 1 - float(running_corrects)/N_train
    top3error = 1 - float(runnning_topk_corrects)/N_train
    epoch_loss = running_loss/N_train

    print('\n| Training loss %.4f\tTop1error %.4f \tTop3error: %.4f'\
            % (epoch_loss, top1error, top3error))

    print_eta(t0, epoch, args.num_epochs)

    ## Validation
    print("Ready for Validation: ", j)
    # Validation 
    running_loss, running_corrects, tot = 0.0, 0.0, 0.0
    runnning_topk_corrects = 0
    torch.set_grad_enabled(False)
    model.eval()
    for batch_idx, (inputs, labels) in enumerate(valset_loaders):
        inputs = cvt_to_gpu(inputs)
        labels = cvt_to_gpu(labels)
        outputs = model(inputs)
        _, preds  = torch.max(outputs.data, 1)
        _, tmpVallabel = torch.max(labels.data, 1)

        top3correct, top3error = mytopk(outputs.data.cpu().numpy(), tmpVallabel, KTOP)
        runnning_topk_corrects += top3correct
        running_loss += loss.item()
        running_corrects += preds.eq(tmpVallabel).cpu().sum()
        tot += labels.size(0)

    epoch_loss = running_loss / N_valid 
    top1error = 1 - float(running_corrects)/N_valid
    ValidationTop1Error[j] = top1error
    top3error = 1 - float(runnning_topk_corrects)/N_valid
    ValidationTop5Error[j] = top3error

    if j == 9:
        top1error = np.mean(ValidationTop1Error)
        top3error = np.mean(ValidationTop5Error)
        print('| Validation loss %.4f\tTop1error %.4f \tTop3error: %.4f \tBestTop3error: %.4f at best epoch: %.4f'\
                % (epoch_loss, top1error, top3error, best_top3, best_epoch))

    ################### save model based on best top3 error
        if top3error < best_top3:
                print('Saving model')
                best_top3 = top3error
                best_epoch = epoch
                best_model = copy.deepcopy(model)
                state = {
                        'model': best_model,
                        'top3' : best_top3,
                        'args': args
                }
                if not os.path.isdir('checkpoint'):
                        os.mkdir('checkpoint')
                save_point = './checkpoint/'
                if not os.path.isdir(save_point):
                        os.mkdir(save_point)

                torch.save(state, save_point + saved_model_fn + '.t7')
                print('=======================================================================')
                print('model saved to %s' % (save_point + saved_model_fn + '.t7'))