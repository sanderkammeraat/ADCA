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

print(colnames)

# column (name) to index
c = dict()
for i, colname in enumerate(colnames):
    
    c[str(colname)] = i
#%%

#Now we can slice out the data like so:
particle_numbers = data[:,c["particle"]]

frame_numbers = data[:, c["frame"]]

#%%

# To make analysis fast in Python we need to use vectorized operations.
# This in turn means that we need to store the data as a numpy array.
# Since numpy arrays have fixed sized, but we have variable numbers of particles,
# we can use masked arrays:  https://numpy.org/doc/stable/reference/maskedarray.html 
# To establish these, we have to find the maximum dimensions.

max_frame_number = int(np.max(frame_numbers))
min_frame_number = int(np.min(frame_numbers))


max_particle_number = int(np.max(particle_numbers))
min_particle_number = int(np.min(particle_numbers))

Nparticles_max = max_particle_number - min_particle_number + 1

Nframes = max_frame_number - min_frame_number + 1

#%%


# Set all values to be masked (True). Upon assignment, they will be unmasked.
x = np.ma.masked_array( np.zeros( shape = (Nframes,Nparticles_max)), True )

y = np.ma.masked_array( np.zeros( shape = (Nframes,Nparticles_max)), True )


#%%

# Let's now fill the arrays, by looping over time.

for i  in range(Nframes):

    #boolean to slice out data that belongs to frame i
    frame_bools = data[:,c["frame"]]==i
    
    frame_i_data = data[frame_bools,:]
    
    frame_i_particle_inds = frame_i_data[:,c["particle"]].astype(int)
    
    
    
    x[i, frame_i_particle_inds] = frame_i_data[:,c["x"]]
    
    y[i, frame_i_particle_inds] = frame_i_data[:,c["y"]]
    #to correctly place the particle data
    #particle_inds = 
    



#%%
fig, ax = plt.subplots()


plt.plot(x[:,0], y[:,0])

plt.show()








