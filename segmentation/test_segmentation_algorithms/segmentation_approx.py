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


fig, ax = plt.subplots()


plt.imshow(im,cmap="Grays_r")


plt.xlabel("x")
plt.ylabel("y")
plt.show()

#%%

imB = cv2.bilateralFilter(im,11,11,11)


#%%
fig, ax = plt.subplots()


plt.imshow(im,cmap="Grays_r")


plt.xlabel("x")
plt.ylabel("y")
plt.show()

#%%

mser = cv2.MSER_create(min_area=200, max_area = 350)
regions, _ = mser.detectRegions(im)

imM = np.zeros(im.shape, dtype="uint8")

for region in regions:
    
    for pixel in region:
    
        imM[pixel[1],pixel[0]]=255
    
    
#%%

fig, ax = plt.subplots()

plt.imshow(im, cmap="Grays_r")
plt.imshow(imM,cmap="Blues", alpha=0.5)


plt.xlabel("x")
plt.ylabel("y")
plt.show()


#%% Close hoes the results aggressively


kernel = np.ones(shape=(3,3), dtype="uint8")

imE = cv2.erode(imM,kernel,iterations = 2)


fig, ax = plt.subplots()
    

plt.imshow(im, cmap="Grays_r")
plt.imshow(imE, alpha=0.5, cmap="Blues")

plt.xlabel("x")
plt.ylabel("y")
plt.show()








#%%


imF = np.zeros(imM.shape, dtype="uint8")


bbox_lims = [0,10000]

area_lims = [200,10000]
(totalLabels, label_ids, values, centroids_all) = cv2.connectedComponentsWithStats(imM)

centr_keep = []

contours_keep = []
for i in range(1, totalLabels): 
    area = values[i, cv2.CC_STAT_AREA]   
    
    height_bounding_box = values[i, cv2.CC_STAT_HEIGHT] 
    
    width_bounding_box = values[i, cv2.CC_STAT_WIDTH] 
    
  
    if (area_lims[0]< area < area_lims[1]) and (bbox_lims[0]<height_bounding_box<bbox_lims[1]) and (bbox_lims[0]<width_bounding_box<bbox_lims[1]) : 
        componentMask = (label_ids == i).astype("uint8") * 255
        
        
        #calculate the contour properties
        
        contours, hierarchy = cv2.findContours(componentMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        
        
        
        cvx = cv2.convexHull(contours[0])
        
        #plt.imshow(cv2.drawContours(im, contours, -1, (0, 255, 0),1) , alpha=0.5, cmap="Blues")

        perimeter = cv2.arcLength(cvx, True)
        
        cvx_area = cv2.contourArea(cvx,True)
        
        p0 = perimeter/np.sqrt(cvx_area)
        
        
        rect = cv2.minAreaRect(contours[0])
        
        
        w = rect[1][0]
        h = rect[1][1]
        
        (x,y),radius = cv2.minEnclosingCircle(contours[0])
        center = (int(x),int(y))
        radius = int(radius)
        
        
        if 10<radius<30:
            

            imF = cv2.bitwise_or(imF, componentMask) 
            centr_keep.append(i)
        


fig, ax = plt.subplots()
    

plt.imshow(im, cmap="Grays_r")
plt.imshow(imF, alpha=0.5, cmap="Blues")

plt.xlabel("x")
plt.ylabel("y")
plt.show()


#%%


#%%What works better, edge detection vs binarization: edge detection gives sharper separations



plt.figure()
#plt.imshow(im, cmap="Grays_r")

plt.imshow(edges, alpha=0.5, cmap="Grays")

plt.imshow(imT, alpha=0.5, cmap="Blues")

plt.xlabel("x")
plt.ylabel("y")
plt.show() 

#%%
kernel = np.ones(shape=(3,3)).astype("uint8")



imD = cv2.dilate(edges,kernel,iterations = 1)


plt.figure()
#plt.imshow(im, cmap="Grays_r")

plt.imshow(imD, alpha=0.5, cmap="Grays")

plt.imshow(imT, alpha=0.5, cmap="Blues")

plt.xlabel("x")
plt.ylabel("y")
plt.show() 

#%%
plt.figure()
#plt.imshow(im, cmap="Grays_r")

plt.imshow(~imD, alpha=0.5, cmap="Grays")


plt.xlabel("x")
plt.ylabel("y")
plt.show() 

#%%
kernel = np.ones(shape=(3,3)).astype("uint8")

imC = cv2.morphologyEx(~imD, cv2.MORPH_OPEN, kernel, iterations=1)


plt.figure()
#plt.imshow(im, cmap="Grays_r")

plt.imshow(imC, alpha=0.5, cmap="Grays")

plt.xlabel("x")
plt.ylabel("y")
plt.show()
#%%




#%%
imF = np.zeros(imD.shape, dtype="uint8")


bbox_lims = [10,40]

area_lims = [100,400]
(totalLabels, label_ids, values, centroids_all) = cv2.connectedComponentsWithStats(~imD)

centr_keep = []

contours_keep = []

for i in range(1, totalLabels): 
    area = values[i, cv2.CC_STAT_AREA]   
    
    height_bounding_box = values[i, cv2.CC_STAT_HEIGHT] 
    
    width_bounding_box = values[i, cv2.CC_STAT_WIDTH] 
    
  
    if (area_lims[0]< area < area_lims[1]) and (bbox_lims[0]<height_bounding_box<bbox_lims[1]) and (bbox_lims[0]<width_bounding_box<bbox_lims[1]) : 
        componentMask = (label_ids == i).astype("uint8") * 255
        
        
        #calculate the contour properties
        
        #contours, hierarchy = cv2.findContours(componentMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        
        
        # plt.imshow(cv2.drawContours(im, contours, -1, (0, 255, 0),1) , alpha=0.5, cmap="Blues")


        
        
        

          
        # # Creating the Final output mask 
        imF = cv2.bitwise_or(imF, componentMask) 
        centr_keep.append(i)
        


fig, ax = plt.subplots()
    

plt.imshow(im, cmap="Grays_r")
plt.imshow(imF, alpha=0.5, cmap="Blues")

plt.xlabel("x")
plt.ylabel("y")
plt.show()
   


#%%


plt.figure()
#plt.imshow(im, cmap="Grays_r")

plt.imshow(~imD)

plt.xlabel("x")
plt.ylabel("y")
plt.show()

#%%


kernel = np.ones(shape=(5,5)).astype("uint8")

imC = cv2.morphologyEx(~imD, cv2.MORPH_OPEN, kernel)


plt.figure()
#plt.imshow(im, cmap="Grays_r")

plt.imshow(imC)

plt.xlabel("x")
plt.ylabel("y")
plt.show()


#%%

imF = np.zeros(imD.shape, dtype="uint8")


bbox_lims = [10,40]

area_lims = [100,400]
(totalLabels, label_ids, values, centroids_all) = cv2.connectedComponentsWithStats(~imT)

centr_keep = []

contours_keep = []

for i in range(1, totalLabels): 
    area = values[i, cv2.CC_STAT_AREA]   
    
    height_bounding_box = values[i, cv2.CC_STAT_HEIGHT] 
    
    width_bounding_box = values[i, cv2.CC_STAT_WIDTH] 
    
  
    if (area_lims[0]< area < area_lims[1]) and (bbox_lims[0]<height_bounding_box<bbox_lims[1]) and (bbox_lims[0]<width_bounding_box<bbox_lims[1]) : 
        componentMask = (label_ids == i).astype("uint8") * 255
        
        
        #calculate the contour properties
        
        #contours, hierarchy = cv2.findContours(componentMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        
        
        # plt.imshow(cv2.drawContours(im, contours, -1, (0, 255, 0),1) , alpha=0.5, cmap="Blues")


        
        
        

          
        # # Creating the Final output mask 
        imF = cv2.bitwise_or(imF, componentMask) 
        centr_keep.append(i)
        


fig, ax = plt.subplots()
    

plt.imshow(im, cmap="Grays_r")
plt.imshow(imF, alpha=0.5, cmap="Blues")

plt.xlabel("x")
plt.ylabel("y")
plt.show()
#%%







    

plt.imshow(im, cmap="Grays_r")
plt.imshow(cv2.drawContours(im, contours_keep[2], -1, (0, 255, 0),1) , alpha=0.5, cmap="Blues")

plt.xlabel("x")
plt.ylabel("y")
plt.show()














