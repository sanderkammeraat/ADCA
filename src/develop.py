#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 16:43:44 2025

@author: kammeraat
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import glob
from tqdm import tqdm
import copy

import addcopyfighandler


#%% < -  These mark cells


#%%
#Get the folder of this project. Using os.path.join() so it works well on Windows and Unix 
project_folder = os.getcwd()

test_trajectories_file_path = os.path.join(project_folder, "testdata","0304","003","Trayectory Hydrazine 003.csv")

test_trajectories_file_path = os.path.join(project_folder, "testdata","0404","003","Trayectory Hydrazine 003_entire_field.csv")

#test_trajectories_file_path = os.path.join(project_folder, "testdata","0404","017","Trayectory Hydrazine 017.csv")



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

for i  in tqdm(range(Nframes)):

    #boolean to slice out data that belongs to frame i
    frame_bools = data[:,c["frame"]]==i
    
    frame_i_data = data[frame_bools,:]
    
    frame_i_particle_inds = frame_i_data[:,c["particle"]].astype(int)
    
    
    
    x[i, frame_i_particle_inds] = frame_i_data[:,c["x"]]
    
    y[i, frame_i_particle_inds] = frame_i_data[:,c["y"]]
    #to correctly place the particle data
    #particle_inds = 
    
#%%

#Calculate the frame displacement vectors 
vx = np.ma.diff(x, axis=0)

vy = np.ma.diff(y, axis=0)
    

#%%

# To test the extraction, plot on top of actual pngs.

frame_folder_path = os.path.join(project_folder, "testdata","0304","003","frames")

frame_folder_path = os.path.join(project_folder, "testdata","0404","003","frames")

#frame_folder_path = os.path.join(project_folder, "testdata","0404","017","frames")

frame_file_paths = sorted(glob.glob(os.path.join(frame_folder_path, "*.png") ))

#%%

#Load frame 0

frame = plt.imread(frame_file_paths[0])


colors = plt.cm.rainbow(np.linspace(0,1,Nparticles_max))



fig, ax = plt.subplots()

title = ax.text(0.5,0.85, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=ax.transAxes, ha="center")



scatter = ax.scatter(x[0,:], y[0,:], c =colors, s=5)

lines = [0] * Nparticles_max


for i in range(Nparticles_max):
    lines[i] = plt.plot(x[:1,i], y[:1,i], c = colors[i] )


image = ax.imshow(frame)
ax.set_xlabel('x')
ax.set_xlabel('y')

def update_frame(i):

    
    title.set_text("frame %d" %i)
    xi = x[i,:]
    yi = y[i,:]
    
    scatter.set_offsets(np.stack([xi,yi]).T)
    
    for j, line in enumerate(lines):
      line[0].set_data(x[:i,j], y[:i,j])
        
    image.set_data(plt.imread(frame_file_paths[i]))
        
    return scatter,  image, title, *[line[0] for line in lines]


anim = animation.FuncAnimation(fig, update_frame, range(0, Nframes, 10), interval=0.1, blit=True, repeat=False)
plt.show()

## Check!


#%%


def detect_clusters(x,y, r_cut):
    
    in_clusters = np.zeros(shape = x.shape,dtype=bool)
    
    
    
    for i in tqdm(range(x.shape[0])):
        xi = x[i,:]
        yi = y[i,:]
        
        for p in range(x.shape[1]):
            
            dxi_p = xi - xi[p]
            
            dyi_p = yi - yi[p]
            
            r2_p = dxi_p**2 + dyi_p**2
        
            
            
            if np.ma.min(r2_p[r2_p>0])<=r_cut**2:
                
                #print(np.ma.min(r2_p[r2_p>0]))
                
                in_clusters[i,p]=True
                
    xc = copy.deepcopy(x)
    yc = copy.deepcopy(y)
    
    #Mask if not in cluster
    xc.mask[~in_clusters]=True
    
    yc.mask[~in_clusters]=True
    
    
    return in_clusters, xc, yc


in_clusters, xc, yc = detect_clusters(x, y, 30)

#%%
            
            
            
            
frame = plt.imread(frame_file_paths[0])


frame = plt.imread(frame_file_paths[0])


colors = plt.cm.rainbow(np.linspace(0,1,Nparticles_max))



fig, ax = plt.subplots(figsize=(20,20))

title = ax.text(0.5,0.85, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=ax.transAxes, ha="center")





lines = [0] * Nparticles_max


for i in range(Nparticles_max):
    lines[i] = plt.plot(xc[:1,i], yc[:1,i], c = colors[i], linewidth = 1)
scatter = ax.scatter(x[0,:].compressed(), y[0,:].compressed(), s=10)

scatter_cluster = ax.scatter(xc[0,:].compressed(), yc[0,:].compressed(), s=50, marker="d", c = "tab:orange")

image = ax.imshow(frame)
ax.set_xlabel('x')
ax.set_xlabel('y')

def update_frame(i):

    
    title.set_text("frame %d" %i)
    xi = x[i,:].compressed()
    yi = y[i,:].compressed()
    
    xic = xc[i,:].compressed()
    yic = yc[i,:].compressed()

    scatter.set_offsets(np.stack([xi,yi]).T)

    scatter_cluster.set_offsets(np.stack([xic,yic]).T)
    
    
    for j, line in enumerate(lines):
      line[0].set_data(x[:i,j], y[:i,j])
        
    image.set_data(plt.imread(frame_file_paths[i]))
        
    return scatter, scatter_cluster, image, title, *[line[0] for line in lines]

plt.tight_layout()
anim = animation.FuncAnimation(fig, update_frame, range(0, Nframes, 20), interval=30, blit=True, repeat=False)



anim.save(filename=os.path.join(project_folder,"analysis_movies", "clusters_0404_003.mp4"), writer="ffmpeg")
plt.show()


#%%


fig, ax = plt.subplots()

i = 1000

image = ax.imshow(plt.imread(frame_file_paths[i]))
ax.scatter(x[i,:].compressed(), y[i,:].compressed())
ax.scatter(xc[i,:].compressed(), yc[i,:].compressed())







        








