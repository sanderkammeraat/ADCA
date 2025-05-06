#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 12:05:19 2025

@author: kammeraat
"""


import segmentation.Segment as sg

import cv2 as cv2

import trackpy as tp

#Testing
if __name__ == "__main__":
    import os
    project_folder = os.getcwd()
    frame_file_path = os.path.join(project_folder,"Hydrazine 003 cropped.tif")
    
    
    #1 is to load it as grayscale image
    
    ims = cv2.imreadmulti(frame_file_path)
    
    frame_array = []
    
    for i in range(12):
        frame_array.append(ims[1][i])
        
    detection = sg.detect_shapes(frame_array, plot_result=True, plot_all_steps=False)
    
    t = tp.link(detection,9)