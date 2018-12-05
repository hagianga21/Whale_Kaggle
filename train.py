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
parser.add_argument('--model_path', type=str, default = ' ')
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_workers', default=1, type=int)
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

#print('Split data...')
#train_img, val_img, train_labels, val_labels = train_test_split(train_df['Image'], y, test_size=args.val_ratio, random_state=2)
#print("Size of Train set: ", train_img.shape)
#print("Size of Valid set: ", train_labels.shape)
#print(train_img[0])
#print(train_labels[0])

input_size = 224 
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

data_transforms =  transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),  # simple data augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])
'''
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),  # simple data augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ]),
}
'''

#dsets = dict()
train_dataset= WhaleDataset(
    datafolder='../data/train/', 
    datatype='train', 
    df=train_df, 
    transform=data_transforms, 
    y=y)
#dsets['train'] = WhaleDataset(datafolder='../data/train/', filenames=train_img, y=train_labels, transform=data_transforms['train'])
#dsets['val'] = WhaleDataset(datafolder='../data/train/', filenames=val_img, y=val_labels, transform=data_transforms['val'])

dset_loaders = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
'''
dset_loaders = {
    x: torch.utils.data.DataLoader(dsets[x],
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   num_workers=args.num_workers)
    for x in ['train']
}
'''
#print(dset_loaders['train'].shape)

########## 
print('Load model')
saved_model_fn = 'resnet' + '-%s' % (args.depth) + '_' + strftime('%m%d_%H%M')
old_model = './checkpoint/' + 'resnet' + '-%s' % (args.depth) + '_' + args.model_path + '.t7'
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
#criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()
model, optimizer = net_frozen(args, model)
model = parallelize_model(model)

N_train = len(train_labels)
#N_valid = len(val_labels)
best_top3 = 1 
t0 = time()

for epoch in range(args.num_epochs):
    optimizer, lr = exp_lr_scheduler(args, optimizer, epoch) 
    print('#################################################################')
    print('=> Training Epoch #%d, LR=%.10f' % (epoch + 1, lr))
    # torch.set_grad_enabled(True)

    running_loss, running_corrects, tot = 0.0, 0.0, 0.0
    running_loss_src, running_corrects_src, tot_src = 0.0, 0.0, 0.0
    runnning_topk_corrects = 0.0
    ########################
    model.train()
    torch.set_grad_enabled(True)
    ## Training 
    # local_src_data = None
    for batch_idx, (inputs, labels) in enumerate(dset_loaders):
        optimizer.zero_grad()
        inputs = cvt_to_gpu(inputs)
        labels = cvt_to_gpu(labels)
        outputs = model(inputs)
        #print(outputs.shape) 32 x 5005 
        #print(labels.shape) 32 x 5005
        loss = criterion(outputs, labels.float())
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
        '''
        sys.stdout.write('| Epoch [%2d/%2d] Iter [%3d/%3d]\tBatch loss %.4f\tTop1error %.4f \tTop3error %.4f'
                         % (epoch + 1, args.num_epochs, batch_idx + 1,
                            (len(train_img) // args.batch_size), batch_loss/args.batch_size,
                            top1error, top3error))
        '''
        sys.stdout.write('| Epoch [%2d/%2d] Iter [%3d/%3d]\tBatch loss %.4f\tTop1error %.4f \tTop3error %.4f\n'
                         % (epoch + 1, args.num_epochs, batch_idx + 1,
                            (len(os.listdir('../data/train')) // args.batch_size), batch_loss/args.batch_size,
                            top1error, top3error))
        sys.stdout.flush()
        sys.stdout.write('\r')

    top1error = 1 - float(running_corrects)/N_train
    top3error = 1 - float(runnning_topk_corrects)/N_train
    epoch_loss = running_loss/N_train
    '''
    print('\n| Training loss %.4f\tTop1error %.4f \tTop3error: %.4f'\
            % (epoch_loss, top1error, top3error))
    '''
    print('\n| Training loss %.4f\tTop1error %.4f \tTop3error: %.4f'\
            % (epoch_loss, top1error, top3error))

    print_eta(t0, epoch, args.num_epochs)

    ###################################
    ## SAVE MODEL
    if top3error < best_top3:
        print('Saving model')
        best_top3 = top3error
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

    '''
    ## Validation
    if (epoch + 1) % args.check_after == 0:
        # Validation 
        running_loss, running_corrects, tot = 0.0, 0.0, 0.0
        runnning_topk_corrects = 0
        torch.set_grad_enabled(False)
        model.eval()
        for batch_idx, (inputs, labels, _) in enumerate(dset_loaders['val']):
            inputs = cvt_to_gpu(inputs)
            labels = cvt_to_gpu(labels)
            outputs = model(inputs)
            _, preds  = torch.max(outputs.data, 1)
            top3correct, top3error = mytopk(outputs.data.cpu().numpy(), labels, KTOP)
            runnning_topk_corrects += top3correct
            running_loss += loss.item()
            running_corrects += preds.eq(labels.data).cpu().sum()
            tot += labels.size(0)

        epoch_loss = running_loss / N_valid 
        top1error = 1 - float(running_corrects)/N_valid
        top3error = 1 - float(runnning_topk_corrects)/N_valid
        print('| Validation loss %.4f\tTop1error %.4f \tTop3error: %.4f'\
                % (epoch_loss, top1error, top3error))

        ################### save model based on best top3 error
        if top3error < best_top3:
            print('Saving model')
            best_top3 = top3error
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
'''