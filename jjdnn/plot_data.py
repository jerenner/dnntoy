# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 2016

Uses code from TensorFlow tutorial:
https://www.tensorflow.org/versions/0.6.0/tutorials/mnist/pros/index.html

@author: jrenner

"""

import sys
import os
import h5py
import numpy as np
import random as rd
from math import *

import scipy.integrate as integrate
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
import matplotlib.projections as prj

from mpl_toolkits.mplot3d import Axes3D
import logging


grdcol = 0.98;
plt_units = "mm";
plt_imgtype = "png";
plt_show = True;
plt_print = True;


fdir = "/Users/jjgomezcadenas/Dropbox/2016/DNN/DATA/h5/"
trk_outdir = "/Users/jjgomezcadenas/Dropbox/2016/DNN/DATA/tracks"


fname_train_bg = fdir+"vox_dnn3d_1mm_bg_combined.h5"
fname_train_si = fdir+"vox_dnn3d_1mm_si_combined.h5"
trk_name = "dnn3d_1mm_si"

fnb_trk = "{0}/{1}".format(trk_outdir,trk_name)
    
# Input variables
vox_ext = 112
vox_size = 1
nclass = 2
num_tracks = 10
trk_startnum=1


# Calculated parameters
pdim = int(2 * vox_ext / vox_size)
npix = pdim * pdim
print "Found dim of {0} for {1} pixels.".format(pdim,npix)

# -----------------------------------------------------------------------------
# Read  the data.
# -----------------------------------------------------------------------------

# Training, signal
h5f_train_si = h5py.File(fname_train_si,'r')

# Create num_tracks tracks.

for ntrk in range(num_tracks):
    
    logging.debug("-- Plotting voxelized track {0}\n".format(ntrk + trk_startnum));

    # Read in the track.
    trkmat = h5f_train_si['trk{0}'.format(ntrk + trk_startnum)]
    varr_x = trkmat[0]
    varr_y = trkmat[1]
    varr_z = trkmat[2]
    varr_c = trkmat[3]*1000
        
    # Plot the 3D voxelized track.
    fig = plt.figure(1)
    fig.set_figheight(9.0)
    fig.set_figwidth(9.0)

    ax1 = fig.add_subplot(111,projection='3d')
    s1 = ax1.scatter(varr_x,varr_y,varr_z,marker='s',
      s=vox_size,linewidth=0.0,c=varr_c,
      cmap=plt.get_cmap('gray_r'),vmin=0.0,vmax=max(varr_c))

    s1.set_edgecolors = s1.set_facecolors = lambda *args:None  
    # this disables automatic setting of alpha relative of distance to camera
    
    ax1.set_xlim([0, 2 * vox_ext]);
    ax1.set_ylim([0, 2 * vox_ext]);
    ax1.set_zlim([0, 2 * vox_ext]);
    ax1.set_xlabel("x (mm)");
    ax1.set_ylabel("y (mm)");
    ax1.set_zlabel("z (mm)");
    ax1.set_title("");

    lb_x = ax1.get_xticklabels();
    lb_y = ax1.get_yticklabels();
    lb_z = ax1.get_zticklabels();
    for lb in (lb_x + lb_y + lb_z):
        lb.set_fontsize(8);

    ax1.w_xaxis.set_pane_color((1.0,1.0,1.0,1.0));
    ax1.w_yaxis.set_pane_color((1.0,1.0,1.0,1.0));
    ax1.w_zaxis.set_pane_color((1.0,1.0,1.0,1.0));
    ax1.w_xaxis._axinfo.update({'grid' : {'color': (grdcol, grdcol, grdcol, 1)}})
    ax1.w_yaxis._axinfo.update({'grid' : {'color': (grdcol, grdcol, grdcol, 1)}})
    ax1.w_zaxis._axinfo.update({'grid' : {'color': (grdcol, grdcol, grdcol, 1)}})

    cb1 = plt.colorbar(s1);
    cb1.set_label('Energy (keV)');
    
    

    # Show and/or print the plot.
    if(plt_print):
        fn_plt = "{0}/plt_{1}_{2}.{3}".format(trk_outdir,
          trk_name,ntrk + trk_startnum,plt_imgtype)
        plt.savefig(fn_plt, bbox_inches='tight')
    if(plt_show):
        plt.show()
        
    plt.close()

