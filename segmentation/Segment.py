#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 11:05:13 2025

@author: kammeraat
"""


import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
import pandas as pd

import trackpy as tp




#Default settings are for triangles, "Hydrazine 003 cropped.tif"

#Assuming  of the form 
#frame_array = nd2.imread(filename,dask=True)

# If you want to check only one frame, say frame i, use detect_spahes([frame_array[i]] )

def detect_shapes(frame_array, p0_min=4.1, p0_max=4.7, area_min=150, area_max=300,  Canny_min=60, Canny_max=100, plot_result=False, plot_all_steps=False):
    
    
    dfs = []
    
    for (fi, frame) in enumerate(frame_array):
        
        print(fi)
    
    
        #Original image
        if plot_all_steps:
            fig, ax = plt.subplots()
            plt.imshow(frame, cmap="Grays_r")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("frame %d, original image (step 0)" %fi)
            plt.show()
            
        
        #Edge dection
        edges = cv2.Canny(frame,Canny_min,Canny_max)
        
        
        if plot_all_steps:
            fig, ax = plt.subplots()                
            plt.imshow(edges)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("frame %d, edge detection (step 1)" %fi)
            plt.show()
    
    
    
        #Dilation to make the inner triangles whole
        kernel = np.ones(shape=(3,3)).astype("uint8")
    
        imD = cv2.dilate(edges,kernel,iterations = 1)
    
        if plot_all_steps:
            plt.figure()
            #plt.imshow(im, cmap="Grays_r")
        
            plt.imshow(imD)
        
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("frame %d, dilation (step 2)" %fi)
            plt.show()
            
            
            
        #Now we use the inner triangle shapes in the inverse of imD, that is, ~imD and extract the  connected componentes in there. We restrict by area and shape index
        imF = np.zeros(imD.shape, dtype="uint8")

        area_lims = [area_min,area_max]
        (totalLabels, label_ids, values, centroids_all) = cv2.connectedComponentsWithStats(~imD)

    
        #The (indices of the centroids) and convex hulls we would like to keep (of the connected components that satisfy the area and shpe index perimenter)
        centr_keep = []


        cvx_hulls = []
        
        #Loop over the connected components and filter on area and shape index
        for i in range(1, totalLabels): 
            area = values[i, cv2.CC_STAT_AREA]   
        
          
            if (area_lims[0]< area < area_lims[1]): 
                componentMask = (label_ids == i).astype("uint8") * 255
                

                
                contours, hierarchy = cv2.findContours(componentMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                 
                cvx = cv2.convexHull(contours[0])
                
                #plt.imshow(cv2.drawContours(im, contours, -1, (0, 255, 0),1) , alpha=0.5, cmap="Blues")
                
                
                
                perimeter = cv2.arcLength(cvx, True)
                
                cvx_area = cv2.contourArea(cvx,True)
                
                p0 = perimeter/np.sqrt(cvx_area)
                

                
                
                if p0_min<p0<p0_max:
                    imF = cv2.bitwise_or(imF, componentMask) 
                    centr_keep.append(i)
                    cvx_hulls.append(cvx)


                    
        centroids = centroids_all[centr_keep]

        x = centroids[:,0]

        y = centroids[:,1]
                    
                    
        if plot_all_steps or plot_result:
            
            fig, ax = plt.subplots()
                
            plt.imshow(cv2.drawContours(frame, cvx_hulls, -1, (0,0,255)), cmap="Blues_r", alpha=0.5, label="convex contours")
            plt.scatter(x,y, label="centroids (com)")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("frame %d, shapes: %d, area & p0 extr. (step 3)" %(fi, len(centroids)))
            plt.legend()
            plt.show()
            
        
        
        
        #Collect results in dataframe and add it to the list of dataframes
        
        df = pd.DataFrame(data={'x':x,'y':y, 'frame':fi})
        
        dfs.append(df)
            
        
    detection = pd.concat(dfs)
    return detection
        


#%%

    
import os
project_folder = os.getcwd()
frame_file_path = os.path.join(project_folder,"segmentation","Hydrazine 003 cropped.tif")


#1 is to load it as grayscale image

ims = cv2.imreadmulti(frame_file_path)

frame_array = []

for i in range(12):
    frame_array.append(ims[1][i])
    
#%%

detection = detect_shapes(frame_array, plot_result=False, plot_all_steps=False)
#%%

t = tp.link(detection,9)


#

