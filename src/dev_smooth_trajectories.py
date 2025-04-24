# -*- coding: utf-8 -*-



import src.Pipeline as pl
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
import scipy as sp
import copy


#%%
project_folder = os.getcwd()

csv_filepath = os.path.join(project_folder, "testdata","0304","003","Trayectory Hydrazine 003.csv")

x, y = pl.load_trajectories(csv_filepath)




#%%

frame_folder_path = os.path.join(project_folder, "testdata","0304","003","frames")

frame_file_paths = sorted(glob.glob(os.path.join(frame_folder_path, "*.png") ))

#Reference frame for plot
fid = 0


fig, ax = plt.subplots()

#Particle id to test smoothing 
pid = 7


plt.imshow(plt.imread(frame_file_paths[fid]))
plt.plot(x[:,pid],y[:,pid], marker='.')


plt.xlabel("x")
plt.ylabel("y")
plt.show()

#%%



def moving_average(x, wlen):
    
    #xs = sp.signal.oaconvolve(x, np.ones( shape = (wlen,1) ), 'valid', axes=0) / wlen
    
    
    
    xs = copy.deepcopy(x [:x.shape[0]-wlen+1,:] )
    for i in range(x.shape[1]):
        
        
        xs[:,i] = np.ma.convolve(x[:,i], np.ones(wlen), "valid")/wlen

    return xs

xs = moving_average(x, 50)

ys = moving_average(y, 50)



fig, ax = plt.subplots()

#Particle id to test smoothing 
pid = 20
    
plt.imshow(plt.imread(frame_file_paths[fid]))
plt.plot(x[:,pid],y[:,pid], marker='.')
plt.plot(xs[:,pid],ys[:,pid], marker='.')


plt.xlabel("x")
plt.ylabel("y")
plt.show()


