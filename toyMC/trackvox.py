"""
trackvox.py

Generates voxelized versions of toyMC tracks.

Attempts a Kalman filter fit on a track using KFTrackFitter

Fit output format:
    
    voxel file: vx vy E
        
    All distances are in mm

"""
import sys,getopt
import numpy as np
import scipy.integrate as integrate
import random as rd
import os
import h5py
from math import *
from trackdefs import *
from scipy.interpolate import interp1d

# -----------------------------------------------------------------------------
# Get the arguments, if any.
# -----------------------------------------------------------------------------
args = sys.argv;
if(len(args) < 1):
    print "Usage:\n\n python trackvox.py <start_event>";
    exit();
trk_startnum = int(args[1]);

# File names.
fnb_trk = "{0}/{1}".format(trk_outdir,trk_name);
vox_outdir = "{0}/{1}".format(trk_outdir,trk_name);

# Directories
if(not os.path.isdir("{0}/{1}".format(trk_outdir,trk_name))):
    print "ERROR: no tracks available in {0}/{1}".format(trk_outdir,trk_name);
    sys.exit();
if(not os.path.isdir(vox_outdir)): os.mkdir(vox_outdir);

# Calculate the length of one dimension of the voxel array.
vdim = int(round(2 * vox_ext / vox_size));

# Open the h5f file containing the events.
h5fr = h5py.File("{0}/{1}_{2}.h5".format(fnb_trk,trk_name,trk_startnum),'r');

# Open an h5f file to save the results.
h5f = h5py.File("{0}/vox_{1}_{2}.h5".format(vox_outdir,trk_name,trk_startnum));

# Create num_tracks tracks.
ntrk = 0
while (ntrk < num_tracks):

    print "-- Track {0} --".format(ntrk + trk_startnum);

    # Read in the track.
    trkmat = h5fr['trk{0}'.format(ntrk + trk_startnum)];
    trk_x0 = trkmat[0];
    trk_y0 = trkmat[1];
    trk_z0 = trkmat[2];
    trk_deltaE = trkmat[3]*1000;

    # Verify that the current grid gives sufficient range.
    rng = 2. * vox_ext;
    xmin = min(trk_x0); ymin = min(trk_y0); zmin = min(trk_z0)
    xmax = max(trk_x0); ymax = max(trk_y0); zmax = max(trk_z0)
    xrng = xmax - xmin; yrng = ymax - ymin; zrng = zmax - zmin
    print "*** Range is: xrng = {0}, yrng = {1}, zrng = {2}".format(xrng,yrng,zrng);
    if(xrng > rng or yrng > rng or zrng > rng):
        print "ERROR: Skipping track with range too high, xrng = {0}, yrng = {1}, zrng = {2}".format(xrng,yrng,zrng);
        exit();
        
    # Calculate the offsets.
    x0 = (xmin + xmax) / 2. - (rng / 2.);
    y0 = (ymin + ymax) / 2. - (rng / 2.);
    z0 = (zmin + zmax) / 2. - (rng / 2.);

    # Create the voxel array.
    if(vox_3d): varr = np.zeros([vdim,vdim,vdim]);
    else: varr = np.zeros([vdim,vdim]);

    # Iterate through the hits, applying offsets and adding their energy to
    #  the appropriate voxels.
    for xhit,yhit,zhit,ehit in zip(trk_x0,trk_y0,trk_z0,trk_deltaE):
        xp = xhit - x0;
        yp = yhit - y0;
        zp = zhit - z0;
        ivox = int(xp / vox_size);
        jvox = int(yp / vox_size);
        kvox = int(zp / vox_size);
        #print "Filling {0},{1},{2}".format(ivox,jvox,kvox)
        if(vox_3d): varr[ivox][jvox][kvox] += ehit;
        else: varr[ivox][jvox] += ehit;
        
    # Get lists of the nonzero x,y,z indices and E values.
    if(vox_3d): nzx,nzy,nzz = np.nonzero(varr);
    else: nzx,nzy = np.nonzero(varr);
    nze = varr[np.nonzero(varr)];
        
    # Combine into one single 4 x N array, where N is the number of nonzero elements.
    if(vox_3d): carr = np.array([nzx, nzy, nzz, nze]);
    else: carr = np.array([nzx, nzy, nze]);
    
    # Save to the h5f file.
    h5f.create_dataset("trk{0}".format(ntrk + trk_startnum),data=carr);
    
    ntrk += 1
    
# Close the h5f files.
h5fr.close();
h5f.close();