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

test_trajectories_file_path = os.path.join(project_folder, "testdata","0404","017","Trayectory Hydrazine 017.csv")



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

frame_folder_path = os.path.join(project_folder, "testdata","0404","017","frames")

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


def detect_clusters(x,y, r_cut, Nframes_stop=None):
    
    in_clusters = np.zeros(shape = x.shape,dtype=bool)
    
    Nframes = x.shape[0]
    if Nframes_stop is not None:
        
        Nframes = Nframes_stop
        
    
    for i in tqdm(range(Nframes)):
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
    
    xnc = copy.deepcopy(x)
    ync = copy.deepcopy(y)
    
    #Mask if not in clusters
    xc.mask[~in_clusters]=True
    
    yc.mask[~in_clusters]=True
    
    xnc.mask[in_clusters]=True
    
    ync.mask[in_clusters]=True
    
    return in_clusters, xc, yc, xnc, ync

# 30 works well for the triangles, 40 for the hexagons
in_clusters, xc, yc, xnc, ync = detect_clusters(x, y, 40)
#%%
vxc = np.ma.diff(xc, axis=0)

vyc = np.ma.diff(yc, axis=0)

vxnc = np.ma.diff(xnc, axis=0)

vync = np.ma.diff(ync, axis=0)

#%%
            
            
            
            
frame = plt.imread(frame_file_paths[0])


frame = plt.imread(frame_file_paths[0])


colors = plt.cm.rainbow(np.linspace(0,1,Nparticles_max))



fig, ax = plt.subplots(figsize=(20,20))

title = ax.text(0.5,0.85, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=ax.transAxes, ha="center")





lines = [0] * Nparticles_max


for i in range(Nparticles_max):
    lines[i] = plt.plot(x[:1,i], y[:1,i], c = colors[i], linewidth = 1)
scatter = ax.scatter(xnc[0,:].compressed(), ync[0,:].compressed(), s=10)

scatter_cluster = ax.scatter(xc[0,:].compressed(), yc[0,:].compressed(), s=50, marker="d", c = "tab:orange")

image = ax.imshow(frame)
ax.set_xlabel('x')
ax.set_xlabel('y')

def update_frame(i):

    
    title.set_text("frame %d" %i)
    xi = xnc[i,:].compressed()
    yi = ync[i,:].compressed()
    
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



anim.save(filename=os.path.join(project_folder,"analysis_movies", "clusters_0404_017.mp4"), writer="ffmpeg")
plt.show()


#%%


fig, ax = plt.subplots()

i = 1000

image = ax.imshow(plt.imread(frame_file_paths[i]))
ax.scatter(x[i,:].compressed(), y[i,:].compressed())
ax.scatter(xc[i,:].compressed(), yc[i,:].compressed())

#%% 

figure_save_folder = os.path.join(project_folder, "testdata","0404","003","plots")

figure_save_folder = os.path.join(project_folder, "testdata","0404","017","plots")





#%%
#Set up units
fps=5
frame2s = 1/fps
pixel2um = 0.32


t = np.arange(Nframes)


vrms = np.ma.sqrt(np.ma.mean(vx**2 + vy**2, axis=1))

vrms_nc = np.ma.sqrt(np.ma.mean(vxnc**2 + vync**2, axis=1))

vrms_c  = np.ma.sqrt(np.ma.mean(vxc**2 + vyc**2, axis=1))



fig, ax = plt.subplots()

ax.plot(t[:-1]*frame2s,vrms_c*pixel2um/frame2s, label="clusters", color="tab:orange")




ax.plot(t[:-1]*frame2s,vrms*pixel2um/frame2s, label="all", color="tab:blue")

ax.plot(t[:-1]*frame2s,vrms_nc*pixel2um/frame2s, label="free", color="tab:green")
ax.set_ylim(0,2)
ax.set_xlabel("t (s)")
ax.set_ylabel("v_rms(t) (um/s)")
ax.legend()

plt.show()
plt.savefig(os.path.join(figure_save_folder,"vrms.pdf"))

#%%
nbins= 10

bin_range = (0,1)


v_hists = [0] * vx.shape[0]

dens = True

for i, v_hist in enumerate(v_hists):
    
    vsi  =np.ma.sqrt(vx[i,:]**2 + vy[i,:]**2)
    
    v_hists[i] = np.histogram( vsi.compressed(),nbins,bin_range, density=dens )
    
vc_hists = [0] * vx.shape[0]

for i, v_hist in enumerate(vc_hists):
    
    vsi  =np.ma.sqrt(vxc[i,:]**2 + vyc[i,:]**2)
    
    vc_hists[i] = np.histogram( vsi.compressed(),nbins, bin_range, density=dens )
    
    
vnc_hists = [0] * vx.shape[0]

for i, v_hist in enumerate(vnc_hists):
    
    vsi  =np.ma.sqrt(vxnc[i,:]**2 + vync[i,:]**2)
    
    vnc_hists[i] = np.histogram( vsi.compressed(),nbins, bin_range, density=dens )
    
    

    

#%%




def edges2centers(bins):

    return (bins[:-1]+bins[1:])/2


def plot_v_over_t_histogram(fig, ax,t, v_hists,title):
    
    t_colors = plt.cm.rainbow(np.linspace(0,1,len(v_hists)))

    
    
    ax.set_title(title)
    
    for i, v_hist in enumerate(v_hists):
        
        bins = v_hist[1]
        
        densities =  v_hist[0]
        
        ax.semilogy( edges2centers(bins)*pixel2um/frame2s ,  densities, c= t_colors[i], label="t= %d s" %(t[i]*frame2s))#, c = t_colors[i])
        
    plt.xlabel("v (um/s)")
    plt.ylabel("p(v)")
    ax.legend()
    plt.show()
    return fig, ax

plot_every = 400

fig, ax = plt.subplots()    
fig, ax = plot_v_over_t_histogram(fig, ax, t[::plot_every],  v_hists[::plot_every], "Distribution of v for all")
plt.show()
plt.savefig(os.path.join(figure_save_folder,"speed_distribution_all.pdf"))

fig, ax = plt.subplots()    
fig, ax = plot_v_over_t_histogram(fig, ax, t[::plot_every],  vnc_hists[::plot_every], "Distribution of v for free")
plt.show()
plt.savefig(os.path.join(figure_save_folder,"speed_distribution_free.pdf"))

fig, ax = plt.subplots()    
fig, ax = plot_v_over_t_histogram(fig, ax, t[::plot_every],  vc_hists[::plot_every], "Distribution of v for clusters")
plt.show()
plt.savefig(os.path.join(figure_save_folder,"speed_distribution_cluster.pdf"))

#%%
nbins= 10

bin_range = (0,1)


v_hists = [0] * vx.shape[0]

dens = True

for i, v_hist in enumerate(v_hists):
    
    vsi  =vx[i,:]
    
    v_hists[i] = np.histogram( vsi.compressed(),nbins,bin_range, density=dens )
    
vc_hists = [0] * vx.shape[0]

for i, v_hist in enumerate(vc_hists):
    
    vsi  =np.ma.sqrt(vxc[i,:]**2 + vyc[i,:]**2)
    
    vc_hists[i] = np.histogram( vsi.compressed(),nbins, bin_range, density=dens )
    
    
vnc_hists = [0] * vx.shape[0]

for i, v_hist in enumerate(vnc_hists):
    
    vsi  =np.ma.sqrt(vxnc[i,:]**2 + vync[i,:]**2)
    
    vnc_hists[i] = np.histogram( vsi.compressed(),nbins, bin_range, density=dens )


#%%
fig, ax = plt.subplots()
n_cluster = np.ma.sum( ~xc.mask, axis=1)

n_all = np.ma.sum( ~x.mask, axis=1)

n_free = np.ma.sum( ~xnc.mask, axis=1)

ax.plot(t*frame2s, n_cluster, label="clusters", color="tab:orange")

ax.plot(t*frame2s, n_free, label="free", color="tab:green")

ax.plot(t*frame2s, n_all, label="all")
ax.legend()

ax.set_ylim(0,1000)

ax.set_xlabel("t (s)")

ax.set_ylabel("N(t)")
plt.show()
plt.savefig(os.path.join(figure_save_folder,"N_over_time.pdf"))

#%%


def calculate_MSD(x, y):
    
    msd = np.ma.masked_array( np.zeros(x.shape[0]) , True )
    ind_0=0
    
    for i in range(x.shape[0]):
        
        msd[i] =  np.ma.mean( (x[i,:] - x[ind_0,:])**2 + (y[i,:] - y[ind_0,:])**2 )


    return msd


msd_nc = calculate_MSD(xnc, ync)

msd_c = calculate_MSD(xc, yc)

msd = calculate_MSD(x, y)

#%%
fig, ax = plt.subplots()

ax.plot(t*frame2s, msd_c*pixel2um**2, label="clusters", color="tab:orange")

ax.plot(t*frame2s, msd_nc*pixel2um**2, label="free", color="tab:green")

ax.plot(t*frame2s, msd*pixel2um**2, label="all", color="tab:blue")

ax.plot(t*frame2s, t**2/40, linestyle='dashed', color="grey", label="slope 2")

ax.set_xlabel("t (s)")

ax.set_ylabel("msd(0,t) (um^2)")

ax.set_xscale("log")
ax.set_yscale("log")
ax.legend()

plt.show()
plt.savefig(os.path.join(figure_save_folder,"MSD.pdf"))

    
    
    





        








