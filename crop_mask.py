# -*- coding: utf-8 -*-
"""
Created on Mon May 24 17:50:06 2021

@author: prajw
"""

from find_mask import masker, masker_np
from utils import trim_masks, to_rle, display, convert_to_mask, crop_by_id
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt

IMAGE_SIZE = 512

with open(".data/label_descriptions.json") as f:
    label_descriptions = json.load(f)

label_names = [x['name'] for x in label_descriptions['categories']]

img, r = masker('s1.jpg')

masks, rois = convert_to_mask(img, r)

    
for idx, i in enumerate(r['class_ids']):
    print(idx, i, label_names[i-2])

# idx = list(r['class_ids']).index(2)
idx = int(input())



res = crop_by_id(img, masks, idx)

display(res)