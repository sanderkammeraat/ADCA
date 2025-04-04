#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 16:43:44 2025

@author: kammeraat
"""

import numpy as np
import matplotlib.pyplot as plt
import os


#%% < -  These mark cells


#%%
#Get the folder of this project. Using os.path.join() so it works well on Windows and Unix 
project_folder = os.getcwd()
test_trajectories_file_path = os.path.join(project_folder, "testdata","Trayectory Hydrazine 003.csv")


data = np.loadtxt(test_trajectories_file_path,delimiter=",", skiprows=1)[:,1:]

colnames =  np.loadtxt(test_trajectories_file_path, dtype=str,delimiter=",", max_rows=1)[1:]


# column (name) to index
c = dict()
for i, colname in enumerate(colnames):
    
    c[str(colname)] = i
#%%


data[:,c["particle"]]


fig, ax = plt.subplots()


# f = 0
# x = data[]
# plt.scatter()






