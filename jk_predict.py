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

import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms.functional as TF
#######################################################################
image_path = '/home/jekim/workspace/resnet_ex/data/images/002600.jpg'
weights_path = '/home/jekim/workspace/resnet_ex/content/cloth_landmarks.pth'
# frontal_face_cascade_path = 'haarcascade_frontalface_default.xml'
#######################################################################
class Network(nn.Module):
    def __init__(self,num_classes=34):
        super().__init__()
        self.model_name='resnet18'
        self.model=models.resnet18(pretrained=False)
        self.model.conv1=nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc=nn.Linear(self.model.fc.in_features,num_classes)
        
    def forward(self, x):
        x=self.model(x)
        return x

#######################################################################
# face_cascade = cv2.CascadeClassifier(frontal_face_cascade_path)

best_network = Network()
best_network.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu'))) 
best_network.eval()

image = cv2.imread(image_path)
# grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height, width,_ = image.shape
h, w,_ = image.shape
new_w=256
new_h=256

# faces = face_cascade.detectMultiScale(grayscale_image, 1.1, 4)

all_landmarks = []
# for (x, y, w, h) in faces:
    # image = grayscale_image[y:y+h, x:x+w]
image = TF.resize(Image.fromarray(image), size=(224, 224))
image = TF.to_tensor(image)
image = TF.normalize(image, [0.5], [0.5])

with torch.no_grad():
    landmarks = best_network(image.unsqueeze(0)) 
    # landmarks = (landmarks.view(17,2).detach().numpy() + 0.5) * np.array([[w, h]]) + np.array([[0, 0]])
    landmarks = (landmarks.view(17,2).detach().numpy())-np.array([w/2,0])
    all_landmarks.append(landmarks)

plt.figure()
plt.imshow(display_image)
for landmarks in all_landmarks:
    plt.scatter(landmarks[:,0], landmarks[:,1], c = 'c', s = 5)

plt.show()