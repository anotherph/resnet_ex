#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 13:53:29 2022

@author: jekim
"""
import os
import sys 
import cv2 as cv
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.models as models

def print_overwrite(step, total_step, loss, operation):
    sys.stdout.write('\r')
    if operation == 'train':
        sys.stdout.write("Train Steps: %d/%d  Loss: %.4f " % (step, total_step, loss))   
    else:
        sys.stdout.write("Valid Steps: %d/%d  Loss: %.4f " % (step, total_step, loss))
        
    sys.stdout.flush()

def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch = \
            sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
                    landmarks_batch[i, :, 1].numpy() + grid_border_size,
                    s=10, marker='.', c='r')

        plt.title('Batch from dataloader')


def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

class ClothLandmarksDataset(Dataset):
    """cloth Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        # return sample
        image = sample['image']
        landmarks = sample['landmarks']
        
        return image, landmarks
    
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
        image, landmarks = sample['image'], sample['landmarks']

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
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


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
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}

class Normalize(object):

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        
        final_img = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX)
        
        return {'image': image, 'landmarks': landmarks}
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}
    
class Network(nn.Module):
    def __init__(self,num_classes=34):
        super().__init__()
        self.model_name='resnet18'
        self.model=models.resnet18()
        self.model.conv1=nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc=nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        x=self.model(x)
        return x
    
    
if __name__ == "__main__":
    
    '''prepare the dataset'''  

    'simple dataset'    
    cloth_dataset = ClothLandmarksDataset(csv_file='data/cloth_landmark.csv',
                                    root_dir='data/images/')
    
    'dataset with tensor and transform'
    
    data_transform = transforms.Compose([
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(),
        Rescale(256),
        RandomCrop(224),
        Normalize(),
        ToTensor()
    ])
    
    cloth_dataset = ClothLandmarksDataset(csv_file='data/cloth_landmark.csv',
                                            root_dir='data/images/',
                                            transform=data_transform)
    
    # cloth_dataset = ClothLandmarksDataset(csv_file='data/cloth_landmark.csv',
    #                                         root_dir='data/images/',
    #                                         transform=transforms.Compose([
    #                                             Rescale(256),
    #                                             RandomCrop(224),
    #                                             ToTensor()
    #                                         ]))
    
    train_dataset,test_dataset=torch.utils.data.random_split(cloth_dataset,(31,10))
    
    train_loader = DataLoader(train_dataset, batch_size=4,shuffle=True, num_workers=0)
    valid_loader = DataLoader(test_dataset, batch_size=4,shuffle=True, num_workers=0)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    ''' train '''
    
    network = Network()
    network.cuda()
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(network.parameters(), lr=0.0001)
    
    loss_min = np.inf
    num_epochs = 10
    
    start_time = time.time()
    for epoch in range(1,num_epochs+1):
        
        loss_train = 0
        loss_valid = 0
        running_loss = 0
        
        network.train()
        for step in range(1,len(train_loader)+1):
        
            images, landmarks = next(iter(train_loader))
            
            images = images.float().cuda()

            landmarks = landmarks.view(landmarks.size(0),-1).float().cuda() 
            
            predictions = network(images)
            
            # clear all the gradients before calculating them
            optimizer.zero_grad()
            
            # find the loss for the current step
            loss_train_step = criterion(predictions, landmarks)
            
            # calculate the gradients
            loss_train_step.backward()
            
            # update the parameters
            optimizer.step()
            
            loss_train += loss_train_step.item()
            running_loss = loss_train/step
            
            print_overwrite(step, len(train_loader), running_loss, 'train')
            
        network.eval() 
        with torch.no_grad():
            
            for step in range(1,len(valid_loader)+1):
                
                images, landmarks = next(iter(valid_loader))
            
                images = images.float().cuda()
                landmarks = landmarks.view(landmarks.size(0),-1).float().cuda()
            
                predictions = network(images)
    
                # find the loss for the current step
                loss_valid_step = criterion(predictions, landmarks)
    
                loss_valid += loss_valid_step.item()
                running_loss = loss_valid/step
    
                print_overwrite(step, len(valid_loader), running_loss, 'valid')
        
        loss_train /= len(train_loader)
        loss_valid /= len(valid_loader)
        
        print('\n--------------------------------------------------')
        print('Epoch: {}  Train Loss: {:.4f}  Valid Loss: {:.4f}'.format(epoch, loss_train, loss_valid))
        print('--------------------------------------------------')
        
        if loss_valid < loss_min:
            loss_min = loss_valid
            # torch.save(network.state_dict(), '/content/face_landmarks.pth') 
            print("\nMinimum Validation Loss of {:.4f} at epoch {}/{}".format(loss_min, epoch, num_epochs))
            print('Model Saved\n')
         
    print('Training Complete')
    print("Total Elapsed Time : {} s".format(time.time()-start_time))
        
    # train(resnet18, params)
    
    '''visulization code'''
    # fig = plt.figure()

    # for i in range(len(cloth_dataset)):
    #     sample = cloth_dataset[i]
    
    #     print(i, sample['image'].shape, sample['landmarks'].shape)
    
    #     ax = plt.subplot(1, 4, i + 1)
    #     plt.tight_layout()
    #     ax.set_title('Sample #{}'.format(i))
    #     ax.axis('off')
    #     show_landmarks(**sample)
    
    #     if i == 3:
    #         plt.show()
    #         break
 
    # for i_batch, sample_batched in enumerate(dataloader_train):
    #     # print(i_batch, sample_batched['image'].size(), sample_batched['landmarks'].size())
    
    #     # observe 4th batch and stop.
    #     if i_batch == 3:
    #         plt.figure()
    #         show_landmarks_batch(sample_batched)
    #         plt.axis('off')
    #         plt.ioff()
    #         plt.show()
    #         break

    # for i in range(train_dataset.__len__()):
    #     # plt.imshow(np.transpose(train_dataset[i][0], (1,2,0)))
    #     plt.imshow(train_dataset[i][0])
    #     plt.show()
    
    # train_dataset[0]
        