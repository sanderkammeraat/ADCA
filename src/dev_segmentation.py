# -*- coding: utf-8 -*-




import matplotlib.pyplot as plt
import matplotlib.animation as animation

import glob
from tqdm import tqdm
import copy
import src.Pipeline as pl


import cv2 as cv
import addcopyfighandler


import os



#%%
project_folder = os.getcwd()



frame_folder_path = os.path.join(project_folder, "testdata","0404","003","frames")

#frame_folder_path = os.path.join(project_folder, "testdata","0404","003","frames")

#frame_folder_path = os.path.join(project_folder, "testdata","0404","017","frames")

frame_file_paths = sorted(glob.glob(os.path.join(frame_folder_path, "*.png") ))

#%%


fig, ax = plt.subplots()

#Particle id to test smoothing 
fid = 20
    
plt.imshow(plt.imread(frame_file_paths[fid]))



plt.xlabel("x")
plt.ylabel("y")
plt.show()



#%%

im = cv.imread(frame_file_paths[fid])



im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
th, im_th = cv.threshold(im_gray, 128, 192, cv.THRESH_OTSU)






fig, ax = plt.subplots()

#Particle id to test smoothing 
fid = 20
    
plt.imshow(im_th, cmap="Grays")



plt.xlabel("x")
plt.ylabel("y")
plt.show()

#%%



imc = cv.bitwise_not(im_th)

fig, ax = plt.subplots()


    
plt.imshow(imc, cmap="Grays")



plt.xlabel("x")
plt.ylabel("y")
plt.show()


#%%





contours, hierarchy = cv.findContours(im_th.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


plt.figure()
plt.imshow(cv.drawContours(im, contours, -1, (0, 255, 0),1) ) 
plt.show()








