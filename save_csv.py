#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 15:06:54 2022
stuff the blank line in cloth landmark

@author: jekim
"""

import os 
import pandas as pd
import numpy as np
     

csv_file='./data/sabok_220330_point_re.csv'
image_dir='./data/image'
landmarks_frame = pd.read_csv(csv_file, header=None) # not to missing first row, add 'header=None'
data_bf=landmarks_frame.to_numpy()
image_list=os.listdir(image_dir)
ind_temp=0;

num_images=len(image_list)
num_keypoints=17
data_af=np.zeros((num_images,num_keypoints*2))
name_list=list()

for ind in range(num_images):
    index=np.where(data_bf[:,0]==image_list[ind])
    
    if len(index[0])==0:
        continue
    elif len(index[0])<num_keypoints:
        name_list.append(image_list[ind])
        temp=data_bf[index,1:]
        k_temp=0
        temp1=np.zeros((num_keypoints,2))
        for k in temp[:,:,0][0]:
            temp1[k-1,:]=temp[:,k_temp,1:]
            k_temp=k_temp+1
    else:
        name_list.append(image_list[ind])
        temp1=data_bf[index,2:]

    data_af[ind_temp,:]=temp1.reshape(-1)
    ind_temp=ind_temp+1
    
data_af=data_af[:ind_temp,:]    

dataframe_af=pd.DataFrame()
dataframe_af.insert(0,"name",name_list,True)

for k in range(num_keypoints*2):
    dataframe_af[k]=data_af[:,k]
    
dataframe_af.to_csv('./data/cloth_landmark.csv',header=False, index=False)