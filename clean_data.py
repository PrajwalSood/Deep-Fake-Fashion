# -*- coding: utf-8 -*-
"""
Created on Mon May  3 16:47:54 2021

@author: prajw
"""

import pandas as pd
from pathlib import Path
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tqdm


DATA_DIR = Path('.data')
ROOT_DIR = Path('root')

with open(DATA_DIR/"label_descriptions.json") as f:
    label_descriptions = json.load(f)

label_names = [x['name'] for x in label_descriptions['categories']]

segment_df = pd.read_csv(DATA_DIR/"train.csv")

multilabel_percent = len(segment_df[segment_df['ClassId'].str.contains('_')])/len(segment_df)*100
print(f"Segments that have attributes: {multilabel_percent:.2f}%")

segment_df['CategoryId'] = segment_df['ClassId'].str.split('_').str[0]

print("Total segments: ", len(segment_df))

sample = segment_df.head(10)


#%% Read Images

image_df = segment_df.groupby('ImageId')['EncodedPixels', 'CategoryId'].agg(lambda x: list(x))


size_df = segment_df.groupby('ImageId')['Height', 'Width'].mean()
image_df = image_df.join(size_df, on='ImageId')

image_df_sample = image_df.head(10)
image_df = image_df.head(1000)

print("Total images: ", len(image_df))

#%%

img_ids = image_df.index

import random as r

enc = {}

for i in range(0,47):
    if i == 0:
        enc.update({i:np.array([255,255,255], dtype = 'uint8')})
    else:
        
        enc.update({i:np.array([r.randint(0,255),r.randint(0,255),r.randint(0,255)], dtype = 'uint8')})

label_enc = pd.DataFrame()
label_enc['Name'] = label_names

label_enc['code'] = list(enc.values())[1:]

label_enc.to_csv('labels.csv')

IMAGE_SIZE = 256
c = 0
for i,row in tqdm.tqdm(image_df.iterrows()):
    name = row.name
    info = image_df.iloc[c]

    c+=1
                
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE, len(info['EncodedPixels'])), dtype=np.uint8)
    labels = []
    
    for m, (annotation, label) in enumerate(zip(info['EncodedPixels'], info['CategoryId'])):
        sub_mask = np.full(info['Height']*info['Width'], 0, dtype=np.uint8)
        annotation = [int(x) for x in annotation.split(' ')]
        if label == '0':
            
            label = '46'
        
        for i, start_pixel in enumerate(annotation[::2]):
            sub_mask[start_pixel: start_pixel+annotation[2*i+1]] = int(label)
    
        sub_mask = sub_mask.reshape((info['Height'], info['Width']), order='F')
        sub_mask = cv2.resize(sub_mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
        
        mask[:, :, m] = sub_mask
        labels.append(int(label)+1)
    
    mask_n = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    mask.shape[2]
    
    for i in range(mask.shape[2]):
        
        y = mask[:,:,i] != 0
        mask_n[y] = mask[:,:,i][y]
    
    
    
    mask_n = np.array([enc[col] for col in list(mask_n.reshape(1,-1)[0])])
    
    mask_n = mask_n.reshape(IMAGE_SIZE, IMAGE_SIZE, 3)
    
    img = cv2.imread('.data/train/' + name)
    plt.imshow(mask_n)
    img = cv2.resize(img, (IMAGE_SIZE,IMAGE_SIZE))
    cv2.imwrite('.data/img/' + name, img)
    plt.imsave('.data/masks/m_' + name, mask_n)
