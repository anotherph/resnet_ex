#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 13:53:29 2022

deepfashion dataset 2 - one-piece 19 landmark detection
add the crop the image alongs to bbox

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
from skimage.color import rgb2gray
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.models as models
import json

class DeepFashion2Dataset(Dataset):
    
    def __init__(self, root_image, root_annos, transform=None):
        """

        """     
        self.root_image = root_image
        self.root_annos = root_annos
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.root_image))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        dir_list_img=os.listdir(self.root_image)
        file_name=os.path.splitext(dir_list_img[idx])[0]
        
        name_img = os.path.join(self.root_image,file_name+'.jpg')
        # image = io.imread(name_img)
        image = rgb2gray(io.imread(name_img))
        
        name_annos = os.path.join(self.root_annos,file_name+'.json')
        with open(name_annos) as f: annos = json.load(f)
        
        landmarks=np.array(annos['item1']['landmarks']).reshape(-1,2)        

        sample = {'image': image, 'landmarks': landmarks}
        
        
        # '''plot'''
        # plt.imshow(image)
        # plt.scatter(landmarks[:,0], landmarks[:,1], c = 'c', s = 5)
        # plt.show()

        if self.transform:
            sample = self.transform(sample)

        # return sample
        image = sample['image']
        landmarks = sample['landmarks'] - 0.5
        
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

class ClothCrop(object):
    " crop the area of the cloth in image"
    
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
        
        
        '''calculate mean & var'''
        transform = transforms.Compose([
            transforms.ToTensor()
            ])
        img_tr = transform(image)
        mean, std = img_tr.mean([1,2]), img_tr.std([1,2])
        
        # mean = torch.Tensor([0.5, 0.5, 0.5])
        # std = torch.Tensor([0.5, 0.5, 0.5])
        
        mean = torch.Tensor([0.5])
        std = torch.Tensor([0.5])
        
        '''normalize the image using mean & var'''
        transform_norm = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean, std)
            ])
        
        img_normalized = transform_norm(image)
        img=np.array(img_normalized)
        
        # image = TF.normalize(image, [0.5], [0.5])
        
        return {'image': img.transpose(1,2,0), 'landmarks': landmarks}
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        landmarks = landmarks / [image.shape[1], image.shape[0]]
        image = image.transpose((2, 0, 1))
        
        # plt.scatter(landmarks[:,0]*image.shape[0], landmarks[:,1]*image.shape[1], c = 'c', s = 5)
        # plt.imshow(image)
        # plt.show()
        
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}
    
class Network(nn.Module):
    def __init__(self,num_classes=19*2):
        super().__init__()
        self.model_name='resnet18'
        self.model=models.resnet18(pretrained=False)
        self.model.conv1=nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc=nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        x=self.model(x)
        return x
    
def print_overwrite(step, total_step, loss, operation):
    sys.stdout.write('\r')
    if operation == 'train':
        sys.stdout.write("Train Steps: %d/%d  Loss: %.4f " % (step, total_step, loss))   
    else:
        sys.stdout.write("Valid Steps: %d/%d  Loss: %.4f " % (step, total_step, loss))
        
    sys.stdout.flush()
    
    
if __name__ == "__main__":
        
    
    train_img_dir = "/home/jekim/workspace/Deepfashion2_Training/Deepfashion2_Training/dataset2_op/train/image"
    train_json_path = "/home/jekim/workspace/Deepfashion2_Training/Deepfashion2_Training/dataset2_op/train/annos"
    valid_img_dir = "/home/jekim/workspace/Deepfashion2_Training/Deepfashion2_Training/dataset2_op/validation/image"
    valid_json_path = "/home/jekim/workspace/Deepfashion2_Training/Deepfashion2_Training/dataset2_op/validation/annos"
    
    data_transform = transforms.Compose([
        Rescale(256),
        RandomCrop(250),
        Normalize(),
        ToTensor()
    ])
    
    dataset_train = DeepFashion2Dataset(root_image=train_img_dir,root_annos=train_json_path,transform=data_transform)
    dataset_valid = DeepFashion2Dataset(root_image=valid_img_dir,root_annos=valid_json_path,transform=data_transform)
        
    a, b=dataset_valid[0]
    
    train_loader = DataLoader(dataset_train, batch_size=16,shuffle=True, num_workers=0)
    valid_loader = DataLoader(dataset_valid, batch_size=4,shuffle=True, num_workers=0)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    ''' train '''
    
    timestr = time.strftime("%Y%m%d_%H%M%S")
    dir_log= os.path.join("/home/jekim/workspace/resnet_ex/log",timestr+'_df2op')
    os.makedirs(dir_log)
    
    network = Network()
    # network.cuda()
    network.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(network.parameters(), lr=0.0001)
    
    loss_min = np.inf
    num_epochs = 50
    
    start_time = time.time()
    
    loss_valid_save = np.array([])
    loss_train_save = np.array([])
    
    for epoch in range(1,num_epochs+1):
        
        loss_train = 0
        loss_valid = 0
        running_loss = 0
        
        network.train()
        for step in range(1,len(train_loader)+1):
        
            images, landmarks = next(iter(train_loader))
            
            images = images.float().cuda()
            # images = images.cuda()

            landmarks = landmarks.view(landmarks.size(0),-1).float().cuda() 
            # landmarks = landmarks.view(landmarks.size(0),-1).cuda() 
            
            # '''visuliaze to check the image and landmarks'''
            # temp=images.cpu().detach().numpy()
            # display_img=np.transpose(temp[0,:,:,:], (1,2,0))
            # temp=landmarks.cpu().detach().numpy()
            # display_landmarks=(temp[0,:].reshape(-1,2)+0.5)           

            # plt.scatter(display_landmarks[:,0]*display_img.shape[0], display_landmarks[:,1]*display_img.shape[1], c = 'c', s = 5)
            # plt.imshow(display_img)
            # plt.show()
            # ''''''
            
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
                # images = images.cuda()
                landmarks = landmarks.view(landmarks.size(0),-1).float().cuda()
                # landmarks = landmarks.view(landmarks.size(0),-1).cuda()
                            
                predictions = network(images)
                
                # '''visuliaze to check the image and landmarks'''
                # temp=images.cpu().detach().numpy()
                # display_img=np.transpose(temp[2,:,:,:], (1,2,0))
                # # temp=landmarks.cpu().detach().numpy()
                # temp =network(images).cpu().detach().numpy()
                # display_landmarks=temp[2,:].reshape(-1,2) + 0.5   

                # plt.scatter(display_landmarks[:,0]*display_img.shape[0], display_landmarks[:,1]*display_img.shape[1], c = 'c', s = 5)
                # plt.imshow(display_img)
                # plt.show()
                # ''''''
    
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
        
        loss_valid_save=np.append(loss_valid_save,loss_valid)
        loss_train_save=np.append(loss_train_save,loss_train)
        
        if loss_valid < loss_min:
            loss_min = loss_valid
            torch.save(network.state_dict(), os.path.join(dir_log,'df2op_'+str(epoch)+'.pth')) 
            print("\nMinimum Validation Loss of {:.4f} at epoch {}/{}".format(loss_min, epoch, num_epochs))
            print('Model Saved\n')
            plt.plot(range(epoch),loss_train_save,'b-o',label='train loss')
            plt.plot(range(epoch),loss_valid_save,'r-o',label='validation loss')
            # legend_without_duplicate_labels(plt)
            # plt.legend()
            plt.grid(True)
            plt.xlabel("epoch")
            plt.ylabel("loss function")
            # plt.show()
            plt.savefig(os.path.join(dir_log,'df2op_loss_function_'+str(epoch)+'.png'), dpi=300)
         
    print('Training Complete')
    print("Total Elapsed Time : {} s".format(time.time()-start_time))
    
    # plt.plot(range(num_epochs),loss_train_save)
    # plt.plot(range(num_epochs),loss_valid_save)
    # plt.savefig('./loss_fuction.png', dpi=300)
    # plt.grid(True)
    # plt.xlabel("epoch")
    # plt.ylabel("loss function")
    # plt.show()

        
        