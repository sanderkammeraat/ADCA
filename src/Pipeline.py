# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm

import copy


def load_trajectories(csv_filepath):
    
    data = np.loadtxt(csv_filepath,delimiter=",", skiprows=1)[:,1:]

    colnames =  np.loadtxt(csv_filepath, dtype=str,delimiter=",", max_rows=1)[1:]

    print(colnames)

    # column (name) to index
    c = dict()
    for i, colname in enumerate(colnames):
        
        c[str(colname)] = i

    #Now we can slice out the data like so:
    particle_numbers = data[:,c["particle"]]

    frame_numbers = data[:, c["frame"]]


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



    # Set all values to be masked (True). Upon assignment, they will be unmasked.
    x = np.ma.masked_array( np.zeros( shape = (Nframes,Nparticles_max)), True )

    y = np.ma.masked_array( np.zeros( shape = (Nframes,Nparticles_max)), True )


    # Let's now fill the arrays, by looping over time.

    for i  in tqdm(range(Nframes)):

        #boolean to slice out data that belongs to frame i
        frame_bools = data[:,c["frame"]]==i
        
        frame_i_data = data[frame_bools,:]
        
        frame_i_particle_inds = frame_i_data[:,c["particle"]].astype(int)
        
        
        
        x[i, frame_i_particle_inds] = frame_i_data[:,c["x"]]
        
        y[i, frame_i_particle_inds] = frame_i_data[:,c["y"]]
    
    return x, y 

def centered_moving_average(x, wlen):
    
    #xs = sp.signal.oaconvolve(x, np.ones( shape = (wlen,1) ), 'valid', axes=0) / wlen
    
    
    
    #xs = copy.deepcopy(x [:x.shape[0]-wlen+1,:] )
    
    xs = copy.deepcopy(x)
    
    xs.mask[:wlen//2,:] = True
    
    xs.mask[-wlen//2,:] = True
    
    
    for i in range(x.shape[1]):
        
        
        xs[wlen//2:-wlen//2+1,i] = np.ma.convolve(x[:,i], np.ones(wlen), "valid")/wlen

    return xs
def smooth_trajectories(x, y, wlen_taxis):
    
    
    
    xs =  centered_moving_average(x, wlen_taxis)
    
    ys = centered_moving_average(y, wlen_taxis)
    
    return xs, ys


