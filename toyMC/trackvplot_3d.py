"""
trackvplot_3d.py

Plots voxelized tracks.

"""
import sys
import numpy as np
import scipy.integrate as integrate
import random as rd
import matplotlib.pyplot as plt
import os
import h5py
from mpl_toolkits.mplot3d import Axes3D
from math import *
from scipy.interpolate import interp1d
from trackdefs import *

from abc import ABCMeta, abstractmethod
import logging 
logging.basicConfig(level=logging.DEBUG)

grdcol = 0.98;

# -----------------------------------------------------------------------------
# Get the arguments, if any.
# -----------------------------------------------------------------------------
args = sys.argv;
if(len(args) < 1):
    print "Usage:\n\n python trackvplot_3d.py <start_event>";
    exit();
trk_startnum = int(args[1]);

fnb_trk = "{0}/{1}".format(trk_outdir,trk_name);

if(not os.path.isdir("{0}/plt".format(fnb_trk))):
    os.mkdir("{0}/plt".format(fnb_trk));
    print "Creating plot directory {0}/plt...".format(fnb_trk);

# Calculate the dimension of the voxel array.
vdim = int(round(2 * vox_ext / vox_size));

# Read in the h5py file.
h5f = h5py.File("{0}/vox_{1}_{2}.h5".format(fnb_trk,trk_name,trk_startnum),'r');

# Create num_tracks tracks.
for ntrk in range(num_tracks):
    
    logging.debug("-- Plotting voxelized track {0}\n".format(ntrk + trk_startnum));

    # Read in the track.
    trkmat = h5f['trk{0}'.format(ntrk + trk_startnum)];
    varr_x = trkmat[0];
    varr_y = trkmat[1];
    varr_z = trkmat[2];
    varr_c = trkmat[3]*1000;
        
    # Plot the 3D voxelized track.
    fig = plt.figure(1);
    fig.set_figheight(5.0);
    fig.set_figwidth(5.0);

    ax1 = fig.add_subplot(111,projection='3d');
    s1 = ax1.scatter(varr_x,varr_y,varr_z,marker='s',s=vox_size,linewidth=0.0,c=varr_c,cmap=plt.get_cmap('gray_r'),vmin=0.0,vmax=max(varr_c));
    s1.set_edgecolors = s1.set_facecolors = lambda *args:None;  # this disables automatic setting of alpha relative of distance to camera
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
    ax1.w_xaxis._axinfo.update({'grid' : {'color': (grdcol, grdcol, grdcol, 1)}});
    ax1.w_yaxis._axinfo.update({'grid' : {'color': (grdcol, grdcol, grdcol, 1)}});
    ax1.w_zaxis._axinfo.update({'grid' : {'color': (grdcol, grdcol, grdcol, 1)}});

    cb1 = plt.colorbar(s1);
    cb1.set_label('Energy (keV)');
        
    # Show and/or print the plot.
    if(plt_print):
        fn_plt = "{0}/plt/plt_{1}_{2}.{3}".format(fnb_trk,trk_name,ntrk + trk_startnum,plt_imgtype);
        plt.savefig(fn_plt, bbox_inches='tight');
    if(plt_show):
        plt.show();
        
    plt.close();