import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class WhaleDataset(Dataset):
    def __init__(self, datafolder, datatype='train', df=None, transform=None, y=None):
        self.datafolder = datafolder
        self.datatype = datatype
        self.y = y
        if self.datatype == 'train':
            self.df = df.values
        self.image_files_list = [s for s in os.listdir(datafolder)]
        self.transform = transform

    def __len__(self):
        return len(self.image_files_list)

    def __getitem__(self, idx):
        if self.datatype == 'train':
            img_name = os.path.join(self.datafolder, self.df[idx][0])
            label = self.y[idx]
        elif self.datatype == 'val':
            img_name = os.path.join(self.datafolder, self.df[idx][0])
            label = self.y[idx]
        elif self.datatype == 'test':
            img_name = os.path.join(self.datafolder, self.image_files_list[idx])
            label = np.zeros((5005,))

        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)
        if self.datatype == 'train':
            return image, label
        elif self.datatype == 'test':
            # so that the images will be in a correct order
            return image, label, self.image_files_list[idx]

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
    y, label_encoder = prepare_labels(train_df['Id'])
    train_dataset = WhaleDataset(datafolder='../data/train/', datatype='train', df=train_df, transform=data_transforms['train'], y=y)
    test_set = WhaleDataset(datafolder='../data/test/', datatype='test', transform=data_transforms['val'])

    print(train_dataset.__len__())