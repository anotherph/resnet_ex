#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 19:06:32 2022

@author: jekim
"""

import json


train_json_path = "/home/jekim/workspace/Deepfashion2_Training/Deepfashion2_Training/dataset2_temp/valid.json"

with open(train_json_path) as f:
    data = json.load(f)

idx=0

temp = data['categories'][idx]['name'] # class 이름
temp1= data['annotations'][idx]['keypoints'] # landmark , find the landmark corresponding to one-piece 