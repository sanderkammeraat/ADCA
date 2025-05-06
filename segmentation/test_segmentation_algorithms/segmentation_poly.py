#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 15:42:23 2025

@author: kammeraat
"""


import matplotlib.pyplot as plt
import matplotlib.animation as animation

import glob
from tqdm import tqdm
import copy
import src.Pipeline as pl


import cv2 as cv2
import addcopyfighandler


import os
import numpy as np
import skimage as ski
import skan

#%%
project_folder = os.getcwd()

frame_file_path = os.path.join(project_folder,"segmentation","Hydrazine 003 cropped.tif")



#1 is to load it as grayscale image

ims = cv2.imreadmulti(frame_file_path)

im = ims[1][1]
#%%

T, imT = cv2.threshold(~im, 0, 255, cv2.THRESH_OTSU)

imski = ski.util.invert(imT)




skeleton = ski.morphology.skeletonize(imski)


#%%

fig, ax = plt.subplots()

    
plt.imshow(skeleton,alpha=0.5,cmap="Blues")
plt.imshow(im,alpha=0.5,cmap="Grays_r")


plt.xlabel("x")
plt.ylabel("y")
plt.show()
#%%

imF = np.zeros(imT.shape, dtype="uint8")


bbox_lims = [5,40]

area_lims = [20,50]
(totalLabels, label_ids, values, centroids_all) = cv2.connectedComponentsWithStats(skeleton.astype("uint8"))

centr_keep = []


label_ids_keep = label_ids.copy()


for i in range(1, totalLabels): 
    area = values[i, cv2.CC_STAT_AREA]   
    
    height_bounding_box = values[i, cv2.CC_STAT_HEIGHT] 
    
    width_bounding_box = values[i, cv2.CC_STAT_WIDTH] 
    
  
    if (area_lims[0]< area < area_lims[1]) and (bbox_lims[0]<height_bounding_box<bbox_lims[1]) and (bbox_lims[0]<width_bounding_box<bbox_lims[1]) : 
        componentMask = (label_ids == i).astype("uint8") * 255
        

          
        # Creating the Final output mask 
        imF = cv2.bitwise_or(imF, componentMask) 
        centr_keep.append(i)
        
    else:
        label_ids_keep[label_ids==i] = 1

fig, ax = plt.subplots()
    

plt.imshow(im, cmap="Grays_r")
plt.imshow(imF, alpha=0.5, cmap="Blues")

plt.xlabel("x")
plt.ylabel("y")
plt.show()

#%%

sobj = skan.csr.Skeleton(imF)




#%%


degs = skan.csr.make_degree_image(imF)

imF2 = imF.copy()

imF2[degs!=3] = 0

plt.figure()
plt.imshow(im, cmap="Grays_r")

plt.imshow(imF2, alpha=0.7, cmap="Reds")
plt.imshow(imF, alpha=0.5, cmap="Blues")

plt.xlabel("x")
plt.ylabel("y")
plt.show()

plt.imshow(L, cmap="Blues_r")



plt.xlabel("x")
plt.ylabel("y")
plt.show()