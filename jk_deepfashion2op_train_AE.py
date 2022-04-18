#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 13:53:29 2022

deepfashion dataset 2 - landmark detection for dataset of sort of one-piece

add the crop the image alongs to bbox

add some variation of transform

model: AE

ground truth : heatmap with landmarks 

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
from scipy.stats import multivariate_normal

class DeepFashion2Dataset(Dataset):
    
    def __init__(self, root_image, root_annos, transform=None):

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
        image = io.imread(name_img)
        # image = cv.imread(name_img, 0)
        # image = rgb2gray(io.imread(name_img))
        
        name_annos = os.path.join(self.root_annos,file_name+'.json')
        with open(name_annos) as f: annos = json.load(f)
        
        landmarks=np.array(annos['item1']['landmarks']).reshape(-1,2)
        
        bbox = np.array(annos['item1']['bounding_box'])

        sample = {'image': image, 'landmarks': landmarks, 'bbox': bbox}
        
        '''plot'''
        # plt.imshow(image)
        # plt.scatter(landmarks[:,0], landmarks[:,1], c = 'c', s = 5)
        # plt.show()

        if self.transform:
            sample = self.transform(sample)

        # return sample
        image = sample['image']
        landmarks = sample['landmarks']
        
        '''make heatmap for ground truth'''
        # heatmap_2dim = np.zeros((image.shape[1],image.shape[2]))
        heatmap = np.zeros((7,image.shape[1],image.shape[2])) # make channel corresponding to the number of class
        length=heatmap.shape[1]
        pos=np.dstack(np.mgrid[0:length:1,0:length:1])
        for int, landmark in enumerate(landmarks):
            rv = multivariate_normal(mean=np.flip(landmark), cov=100)
            heatmap[int,:,:]=rv.pdf(pos)/rv.pdf(pos).max()
        
        # heatmap[:,:,0]=heatmap_2dim
        # heatmap[:,:,1]=heatmap_2dim
        # heatmap[:,:,2]=heatmap_2dim
        
        # heatmap=torch.from_numpy(heatmap_2dim.transpose((2,0,1)))
        
        heatmap=torch.from_numpy(heatmap)
                   
        return image, heatmap
             
class ClothCrop(object):
    " crop the area of the cloth in image"
    
    def __call__(self, sample):
        image, landmarks, bbox = sample['image'], sample['landmarks'], sample['bbox']
    
        '''bbox = [x1,y1,x2,y2]'''        
        
        # image = image[bbox[1]: bbox[3], bbox[0]: bbox[2]]        
        # landmarks = landmarks - [bbox[0], bbox[1]]
        
        '''extended bbox'''
        
        spare=30
        if bbox[3]+spare < image.shape[0] and bbox[2]+spare < image.shape[1] and bbox[1]-spare>0 and bbox[0]-spare>0:
            image = image[bbox[1]-spare: bbox[3]+spare, bbox[0]-spare: bbox[2]+spare]
            landmarks = landmarks - [bbox[0]-spare, bbox[1]-spare]
        else:
            image = image[bbox[1]: bbox[3], bbox[0]: bbox[2]]
            landmarks = landmarks - [bbox[0], bbox[1]]
    
        
        return {'image': image, 'landmarks': landmarks}
    
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
    
class Rescale_padding(object):
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
        
        '''rescale'''

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h < w:
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
        
        '''padding'''
        
        h, w = img.shape[:2]
        # max_wh = np.max([w, h]) #padding
        max_wh = self.output_size+30 #padding to extended box
        h_padding = (max_wh - w) / 2
        v_padding = (max_wh - h) / 2
        l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5 # left
        t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5 # top
        r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5 # right
        b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5 # bottom
        padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
        
        img_padded = np.ones((max_wh,max_wh,3))*(-1)
        img_padded[int(b_pad):int(b_pad)+h,int(l_pad):int(l_pad)+w,:]=img
        
        # img_padded = np.ones((max_wh,max_wh))*(-1)
        # img_padded[int(b_pad):int(b_pad)+h,int(l_pad):int(l_pad)+w]=img
        
        landmarks = landmarks + [int(l_pad),int(b_pad)]
                

        return {'image': img_padded, 'landmarks': landmarks}

class Resize(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        
        image, landmarks = sample['image'], sample['landmarks']
        
        h, w = image.shape[:2]
        
        img = transform.resize(image, [self.output_size,self.output_size])

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [self.output_size / w, self.output_size / h]

        return {'image': img, 'landmarks': landmarks}

class Normalize(object):

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        
        '''normalize the image using mean & var'''
        transform_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # transforms.Normalize([0.5], [0.5])
            ])
        
        img_normalized = transform_norm(image)
        img=np.array(img_normalized)
        
        # landmarks = landmarks / [img.transpose(1,2,0).shape[1], img.transpose(1,2,0).shape[0]]
        
        # image = TF.normalize(image, [0.5], [0.5])
        
        # return {'image': img.transpose(1,2,0), 'landmarks': landmarks}
        
        return {'image': img_normalized, 'landmarks': landmarks}
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        # landmarks = landmarks / [image.shape[1], image.shape[0]]
        image = image.transpose((2, 0, 1))
        
        # plt.scatter(landmarks[:,0]*image.shape[0], landmarks[:,1]*image.shape[1], c = 'c', s = 5)
        # plt.imshow(image)
        # plt.show()
        
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}
    
'''network'''

class AutoEncoder(nn.Module): 
    def __init__(self): 
        super(Autoencoder, self).__init__() 
        self.encoder = nn.Sequential( 
            nn.Linear(28*28, 128), 
            nn.ReLU(), 
            nn.Linear(128, 64), 
            nn.ReLU(), 
            nn.Linear(64, 12), 
            nn.ReLU(), nn.Linear(12, 3), 
            ) 
        
        self.decoder = nn.Sequential( 
            nn.Linear(3, 12), 
            nn.ReLU(), 
            nn.Linear(12, 64), 
            nn.ReLU(), 
            nn.Linear(64, 128), 
            nn.ReLU(), 
            nn.Linear(128, 28*28), 
            nn.Sigmoid(), 
            ) 
        
        def forward(self, x): 
            encoded = self.encoder(x) 
            decoded = self.decoder(encoded) 
            return encoded, decoded

class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()
        
        # Encoder
        self.cnn_layer1 = nn.Sequential(
                        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                         nn.MaxPool2d(2,2))

        self.cnn_layer2 = nn.Sequential(
                                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                 nn.MaxPool2d(2,2))
        
        self.cnn_layer3 = nn.Sequential(
                                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                 nn.MaxPool2d(2,2))

        # Decoder
        
        self.tran_cnn_layer1 = nn.Sequential(
                        nn.ConvTranspose2d(64, 32, kernel_size = 3, stride = 2, padding=0),
                        nn.ReLU())
        
        self.tran_cnn_layer2 = nn.Sequential(
                        nn.ConvTranspose2d(32, 16, kernel_size = 3, stride = 2, padding=0),
                        nn.ReLU())

        self.tran_cnn_layer3 = nn.Sequential(
                        nn.ConvTranspose2d(16, 7, kernel_size = 3, stride = 2, padding=0),
                        nn.Sigmoid())
            
            
    def forward(self, x):
        output = self.cnn_layer1(x)
        output = self.cnn_layer2(output)
        output = self.cnn_layer3(output)
        output = self.tran_cnn_layer1(output)
        output = self.tran_cnn_layer2(output)
        output = self.tran_cnn_layer3(output)

        return output[:,:,:286,:286]
    
def print_overwrite(step, total_step, loss, operation):
    sys.stdout.write('\r')
    if operation == 'train':
        sys.stdout.write("Train Steps: %d/%d  Loss: %.4f " % (step, total_step, loss))   
    else:
        sys.stdout.write("Valid Steps: %d/%d  Loss: %.4f " % (step, total_step, loss))
        
    sys.stdout.flush()
    
    
if __name__ == "__main__":
        
    
    train_img_dir = "/home/jekim/workspace/Deepfashion2_Training/Deepfashion2_Training/dataset2_op_small_lower/train/image"
    train_json_path = "/home/jekim/workspace/Deepfashion2_Training/Deepfashion2_Training/dataset2_op_small_lower/train/annos"
    valid_img_dir = "/home/jekim/workspace/Deepfashion2_Training/Deepfashion2_Training/dataset2_op_small_lower/validation/image"
    valid_json_path = "/home/jekim/workspace/Deepfashion2_Training/Deepfashion2_Training/dataset2_op_small_lower/validation/annos"
    
    data_transform = transforms.Compose([
        ClothCrop(),
        # Resize(256),
        Rescale_padding(256),
        Normalize()
        # ToTensor()
    ])
    
    dataset_train = DeepFashion2Dataset(root_image=train_img_dir,root_annos=train_json_path,transform=data_transform)
    dataset_valid = DeepFashion2Dataset(root_image=valid_img_dir,root_annos=valid_json_path,transform=data_transform)
        
    image, landmarks=dataset_valid[100] # check the data and length of tensor
    
    batch_size= 32 
    train_loader = DataLoader(dataset_train, batch_size=batch_size,shuffle=True, num_workers=0)
    valid_loader = DataLoader(dataset_valid, batch_size=batch_size,shuffle=True, num_workers=0)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    ''' train '''
    
    timestr = time.strftime("%Y%m%d_%H%M%S")
    dir_log= os.path.join("/home/jekim/workspace/resnet_ex/log",timestr+'_df2op')
    os.makedirs(dir_log)
    
    network = ConvAutoEncoder()
    # network = CNN()
    network.to(device)
    
    criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=0.0001)
    # optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
    
    loss_min = np.inf
    num_epochs = 15
    
    start_time = time.time()
    
    loss_valid_save = np.array([])
    loss_train_save = np.array([])
    
    for epoch in range(1,num_epochs+1):
        
        loss_train = 0
        loss_valid = 0
        running_loss = 0
        
        network.train()
        for step in range(1,len(train_loader)+1):
        
            images, heatmap = next(iter(train_loader))
            
            images = images.float().cuda()
            heatmap = heatmap.float().cuda()
            # images = images.cuda()

            # landmarks = landmarks.view(landmarks.size(0),-1).float().cuda() 
            # landmarks = landmarks.view(landmarks.size(0),-1).cuda() 
        
            predictions = network(images)
            
            # '''visuliaze to check the image and landmarks'''
            # temp=images.cpu().detach().numpy()
            # display_img=np.transpose(temp[0,:,:,:], (1,2,0)).squeeze()
            # temp=heatmap.cpu().detach().numpy()
            # display_heatmap=np.transpose(temp[0,:,:,:], (1,2,0)).squeeze()
            # temp=predictions.cpu().detach().numpy()
            # display_result=np.transpose(temp[0,:,:,:], (1,2,0)).squeeze()

            # plt.imshow(display_img+10*display_heatmap)
            # plt.show()
            # ''''''
            
            # clear all the gradients before calculating them
            optimizer.zero_grad()
            
            # find the loss for the current step
            loss_train_step = criterion(predictions, heatmap)
            
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
                
                images, heatmap = next(iter(valid_loader))
            
                images = images.float().cuda()
                heatmap = heatmap.float().cuda()
                # images = images.cuda()
                # landmarks = landmarks.view(landmarks.size(0),-1).float().cuda()
                # landmarks = landmarks.view(landmarks.size(0),-1).cuda()
                            
                predictions = network(images)
                
                # '''visuliaze to check the image and landmarks'''
                # temp=images.cpu().detach().numpy()
                # display_img=np.transpose(temp[0,:,:,:], (1,2,0)).squeeze()
                # temp=heatmap.cpu().detach().numpy()
                # display_heatmap=np.transpose(temp[0,:,:,:], (1,2,0)).squeeze()
                # temp=predictions.cpu().detach().numpy()
                # display_result=np.transpose(temp[0,:,:,:], (1,2,0)).squeeze()

                # plt.imshow(display_img+10*display_result)
                # plt.show()
                # ''''''
    
                # find the loss for the current step
                loss_valid_step = criterion(predictions, heatmap)
    
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
        
        # if loss_valid < loss_min:
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

        
        
