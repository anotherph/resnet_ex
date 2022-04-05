#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 18:49:00 2022

@author: jekim

prediction for jk_train.py
"""

import time
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imutils
from skimage import io, transform

import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class test_img(Dataset):
    """cloth Landmarks dataset."""

    def __init__(self,root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    # def __len__(self):
    #     return len(self.landmarks_frame)

    def __getitem__(self, idx):
    # def __getitem__(self):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # img_name = os.path.join(self.root_dir,
        #                         self.landmarks_frame.iloc[idx, 0])
        img_name = os.path.join(self.root_dir)
        image = io.imread(img_name)
        # landmarks = self.landmarks_frame.iloc[idx, 1:]
        # landmarks = np.array([landmarks])
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        # sample = {'image': image, 'landmarks': landmarks}
        sample = {'image': image}

        if self.transform:
            sample = self.transform(sample)

        # return sample
        image = sample['image']
        # landmarks = sample['landmarks']
        
        # return image, landmarks
        return image

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        # image, landmarks = sample['image'], sample['landmarks']
        image=sample['image']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        # landmarks = landmarks * [new_w / w, new_h / h]

        # return {'image': img, 'landmarks': landmarks}
        return {'image':img}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        # image, landmarks = sample['image'], sample['landmarks']
        image=sample['image']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        # landmarks = landmarks - [left, top]    

        # return {'image': image, 'landmarks': landmarks}
        return {'image':image}

class Normalize(object):

    def __call__(self, sample):
        # image, landmarks = sample['image'], sample['landmarks']
        image=sample['image']
        
        # final_img = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX)
        
        '''calculate mean & var'''
        transform = transforms.Compose([
            transforms.ToTensor()
            ])
        img_tr = transform(image)
        mean, std = img_tr.mean([1,2]), img_tr.std([1,2])
        
        '''normalize the image using mean & var'''
        transform_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])
        
        img_normalized = transform_norm(image)
        img=np.array(img_normalized)
        
        # return {'image': img.transpose(1,2,0), 'landmarks': landmarks}
        return {'image': img.transpose(1,2,0)}
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # image, landmarks = sample['image'], sample['landmarks']
        image=sample['image']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        
        # landmarks = landmarks / [image.shape[1], image.shape[0]]
        image = image.transpose((2, 0, 1))
        
        # return {'image': torch.from_numpy(image),
        #         'landmarks': torch.from_numpy(landmarks)}
        
        return {'image': torch.from_numpy(image)}

    
#######################################################################
image_path = '/home/jekim/workspace/Deepfashion2_Training/Deepfashion2_Training/dataset1_temp/train/img/0107.jpg'
weights_path = '/home/jekim/workspace/resnet_ex/content/cloth_landmarks_coco.pth'
# frontal_face_cascade_path = 'haarcascade_frontalface_default.xml'
#######################################################################
class Network(nn.Module):
    def __init__(self,num_classes=12):
        super().__init__()
        self.model_name='resnet18'
        self.model=models.resnet18(pretrained=False)
        self.model.conv1=nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc=nn.Linear(self.model.fc.in_features,num_classes)
        
    def forward(self, x):
        x=self.model(x)
        return x

#######################################################################

best_network = Network()
best_network.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu'))) 
best_network.eval()

image = cv2.imread(image_path)

all_landmarks = []


data_transform = transforms.Compose([
    Rescale(256),
    RandomCrop(250),
    Normalize(),
    ToTensor()
])

imagetotest=test_img(root_dir=image_path,transform=data_transform)
    

image_display=imagetotest[0].numpy().transpose((1,2,0))
image_test=imagetotest[0].unsqueeze(0).float()

with torch.no_grad():
    landmarks = best_network(image_test)
    landmarks = (landmarks.view(6,2).detach().numpy())
    all_landmarks.append(landmarks)

plt.figure()
plt.imshow(image_display)
plt.scatter(landmarks[:,0]*image_display.shape[0], landmarks[:,1]*image_display.shape[1], c = 'c', s = 5)

plt.show()