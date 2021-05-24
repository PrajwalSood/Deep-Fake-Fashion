# -*- coding: utf-8 -*-
"""
Created on Mon May 24 18:30:22 2021

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

vid = cv2.VideoCapture(0)
  
while(True):
      

    ret, frame = vid.read()

    try:
        img, r = masker_np(frame)
        masks, rois = convert_to_mask(img, r)
        idx = list(r['class_ids']).index(2)
        res = crop_by_id(img, masks, idx)
        cv2.imshow('frame', res)
    except:
        cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()