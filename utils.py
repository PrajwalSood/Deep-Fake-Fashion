# -*- coding: utf-8 -*-
"""
Created on Mon May 24 18:10:09 2021

@author: prajw
"""

import numpy as np
import itertools
import cv2

def to_rle(bits):
    rle = []
    pos = 1
    for bit, group in itertools.groupby(bits):
        group_list = list(group)
        if bit:
            rle.extend([pos, len(group_list)])
        pos += len(group_list)
    return rle


def trim_masks(masks, rois, class_ids):
    class_pos = np.argsort(class_ids)
    class_rle = to_rle(np.sort(class_ids))
    
    pos = 0
    for i, _ in enumerate(class_rle[::2]):
        previous_pos = pos
        pos += class_rle[2*i+1]
        if pos-previous_pos == 1:
            continue 
        mask_indices = class_pos[previous_pos:pos]
        
        union_mask = np.zeros(masks.shape[:-1], dtype=bool)
        for m in mask_indices:
            masks[:, :, m] = np.logical_and(masks[:, :, m], np.logical_not(union_mask))
            union_mask = np.logical_or(masks[:, :, m], union_mask)
        for m in mask_indices:
            mask_pos = np.where(masks[:, :, m]==True)
            if np.any(mask_pos):
                y1, x1 = np.min(mask_pos, axis=1)
                y2, x2 = np.max(mask_pos, axis=1)
                rois[m, :] = [y1, x1, y2, x2]
            
    return masks, rois

def display(img):
    cv2.imshow('window', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def convert_to_mask(img, r, IMAGE_SIZE = 512):
    if r['masks'].size > 0:
        masks = np.zeros((img.shape[0], img.shape[1], r['masks'].shape[-1]), dtype=np.uint8)
        for m in range(r['masks'].shape[-1]):
            masks[:, :, m] = cv2.resize(r['masks'][:, :, m].astype('uint8'), 
                                        (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        y_scale = img.shape[0]/IMAGE_SIZE
        x_scale = img.shape[1]/IMAGE_SIZE
        rois = (r['rois'] * [y_scale, x_scale, y_scale, x_scale]).astype(int)
        
        masks, rois = trim_masks(masks, rois, r['class_ids'])
    else:
        masks, rois = r['masks'], r['rois']
    return masks, rois

def crop_by_id(img, masks, processing):
    res = img.copy()    
    res[...,0] = np.where(masks[:,:,processing], img[:,:,0], masks[:,:,processing])
    res[...,1] = np.where(masks[:,:,processing], img[:,:,1], masks[:,:,processing])
    res[...,2] = np.where(masks[:,:,processing], img[:,:,2], masks[:,:,processing])
    return res