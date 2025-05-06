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


#%%
project_folder = os.getcwd()

frame_file_path = os.path.join(project_folder,"segmentation","Hydrazine 003 cropped.tif")


#1 is to load it as grayscale image

ims = cv2.imreadmulti(frame_file_path)

for i in range(12):
    im = ims[1][i]
    
    
im = ims[1][0]
    #%%
    
fig, ax = plt.subplots()

    
plt.imshow(im)



plt.xlabel("x")
plt.ylabel("y")
plt.show()

#%%


edges = cv2.Canny(im,60,100)

fig, ax = plt.subplots()

    
plt.imshow(edges)



plt.xlabel("x")
plt.ylabel("y")
plt.show()

#%%
kernel = np.ones(shape=(3,3)).astype("uint8")



imD = cv2.dilate(edges,kernel,iterations = 1)


plt.figure()
#plt.imshow(im, cmap="Grays_r")

plt.imshow(imD)

plt.xlabel("x")
plt.ylabel("y")
plt.show()

#%%
kernel = np.ones(shape=(3,3)).astype("uint8")



imD = cv2.dilate(edges,kernel,iterations = 1)


plt.figure()
#plt.imshow(im, cmap="Grays_r")

plt.imshow(~imD)

plt.xlabel("x")
plt.ylabel("y")
plt.show()


#%%

imF = np.zeros(imD.shape, dtype="uint8")



area_lims = [150,300]
(totalLabels, label_ids, values, centroids_all) = cv2.connectedComponentsWithStats(~imD)

centr_keep = []


cvx_hulls = []



for i in range(1, totalLabels): 
    area = values[i, cv2.CC_STAT_AREA]   
    
    height_bounding_box = values[i, cv2.CC_STAT_HEIGHT] 
    
    width_bounding_box = values[i, cv2.CC_STAT_WIDTH] 
    
  
    if (area_lims[0]< area < area_lims[1]): 
        componentMask = (label_ids == i).astype("uint8") * 255
        

        
        contours, hierarchy = cv2.findContours(componentMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

         
        cvx = cv2.convexHull(contours[0])
        
        #plt.imshow(cv2.drawContours(im, contours, -1, (0, 255, 0),1) , alpha=0.5, cmap="Blues")

        perimeter = cv2.arcLength(cvx, True)
        
        cvx_area = cv2.contourArea(cvx,True)
        
        p0 = perimeter/np.sqrt(cvx_area)
        

        
        
        if 4.1<p0<4.7:
            imF = cv2.bitwise_or(imF, componentMask) 
            centr_keep.append(i)
            cvx_hulls.append(cvx)
fig, ax = plt.subplots()
    

#plt.imshow(im, cmap="Grays_r")
#plt.imshow(imF, alpha=0.5, cmap="Blues")
plt.imshow(cv2.drawContours(im, cvx_hulls, -1, (0,255,0)), cmap="Blues_r", alpha=0.5)

plt.xlabel("x")
plt.ylabel("y")
plt.show()


#%%   
centroids = centroids_all[centr_keep]




x = centroids[:,0]

y = centroids[:,1]


fig, ax = plt.subplots()

    
plt.imshow(im, cmap="Grays_r")
plt.scatter(x,y)

plt.xlabel("x")
plt.ylabel("y")
plt.show()

print(len(x))
