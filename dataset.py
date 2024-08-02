import torch
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from utils import *

class Polyp_datasets(Dataset):
    def __init__(self, input_size_h, input_size_w, train=True):
        super(Polyp_datasets, self)
        path_Data = '/content/drive/MyDrive/kvasir'
        if train:
            images_list = os.path.join(path_Data, 'train/img')
            masks_list = os.path.join(path_Data, 'train/mask')
            self.data = []
            for p in os.listdir(images_list):
                name = p.split('.')[0]
                img_path = path_Data+'/train/img/' + name + '.jpg'
                mask_path = path_Data+'/train/mask/' + name + '.jpg'
                self.data.append([img_path, mask_path])
            self.transformer  = transforms.Compose([
                        #myNormalize(datasets, train=True), #
                        myToTensor(),
                        myRandomHorizontalFlip(p=0.5),
                        myRandomVerticalFlip(p=0.5),
                        myRandomRotation(p=0.5, degree=[0, 360]),
                        myResize(input_size_h, input_size_w)
    ])
        else: 
            images_list = os.path.join(path_Data, 'test/img')
            masks_list = os.path.join(path_Data, 'test/mask')
            self.data = []
            for p in os.listdir(images_list):
                name = p.split('.')[0]
                img_path = path_Data+'/test/img/' + name + '.jpg' 
                mask_path = path_Data+'/test/mask/' + name + '.jpg' 
                self.data.append([img_path, mask_path])
            self.transformer = transforms.Compose([
        #myNormalize(datasets, train=False),
        myToTensor(),
        myResize(input_size_h, input_size_w)
    ])
        
    def __getitem__(self, index):
        img_path, msk_path = self.data[index]
        img = np.array(Image.open(img_path).convert('RGB'))
        
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
        img, msk = self.transformer((img, msk))
        return img, msk

    def __len__(self):
        return len(self.data)
    

def prepare_dataset_kvasir(num_clients: int,
                           batch_size: int, 
                           input_size_h: int, 
                           input_size_w: int, 
                           val_ratio: float = 0.1):

    trainset = Polyp_datasets( input_size_h, input_size_w, train=True)
    testset = Polyp_datasets( input_size_h, input_size_w, train=False)

    
    # we split data equal for each client because we assume that dataset is iid

    num_images = len(trainset) // num_clients

    partition_len = [num_images] * num_clients
    trainsets = random_split(trainset, partition_len,torch.Generator().manual_seed(2023))

    trainloaders = []
    valloaders = []
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(
            trainset_, [num_train, num_val], torch.Generator().manual_seed(2023)
        )

        trainloaders.append(
            DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2)
        )
        valloaders.append(
            DataLoader(for_val, batch_size=1, shuffle=False, num_workers=2)
        )

     
    testloader = DataLoader(testset, batch_size=1)

    return trainloaders, valloaders, testloader