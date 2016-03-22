"""
trackdefs.py

Function definitions and constants for toy MC track generation.

"""
import sys
import numpy as np
import scipy.integrate as integrate
import random as rd
import os
import matplotlib.colors as mpcol
from math import *

# -------------------------------------------------------------------------------------------------------
# Frequently modified parameters:
# -------------------------------------------------------------------------------------------------------

# Output directories
trk_outdir = "/Users/jrenner/IFIC/dnn/tracks";
#trk_outdir = "/data4/NEXT/users/jrenner/dnn/dnntoy/tracks";   # output directory for tracks

trk_name = "dnn3d_NEXT100_Paolina222_v2x2x2_r200x200x200_bg";  # name assigned to this run; will be used in naming the output files
trk_bb = True;       # set to true to create "double-beta"-like tracks
trk_genbb = True;     # for double-beta events, use genbb information
num_tracks = 10;  # number of tracks to generate and/or fit

Bfield = 0.0;      # applied magnetic field in Tesla
MS0 = 13.6;        # multiple scattering parameter (should be 13.6)

# Voxelization parameters
vox_ext = 100;     # extent in x,y,z of voxelized track (in +/- directions): full allowed range is 2 * vox_ext
vox_size = 2;      # voxel size
vox_3d = True;    # output 3D voxels or 2D (x-y) projection
vox_already = False;  # set to true if input tracks are already voxelized
vox_prev_size = 2;   # for already voxelized tracks, the size of voxelization

# Energy parameters
E_0 = 2.4578+0.511; # initial energy in MeV
eslice = 0.0025;     # slice energy in MeV
E_tol = 1e-3;      # energy tolerance value (energy is considered to be 0 if less than this value)

# Gas configuration parameters.
Pgas = 15.;       # gas pressure in atm
Tgas = 293.15;    # gas temperature in Kelvin
LrXe = 15300.;     # xenon radiation length  * pressure in mm * bar
Lr = LrXe/(Pgas/1.01325);

# Plot options.
plt_units = "mm";
plt_imgtype = "png";
plt_show = False;
plt_print = True;

# -----------------------------------------------------------------------------
# Less frequently modified parameters:
# -----------------------------------------------------------------------------

# Physics constants.
pc_rho0 = 2.6867774e19;   # density of ideal gas at T=0C, P=1 atm in cm^(-3)
pc_m_Xe = 131.293;        # mass of xenon in amu
pc_NA = 6.02214179e23;    # Avogadro constant
pc_eC = 1.602176487e-19;  # electron charge in C
pc_me = 9.10938215e-31;   # electron mass in kg
pc_me_MeV = 0.511;        # electron mass in MeV
pc_clight = 2.99792458e8;	    # speed of light in m/s

# -----------------------------------------------------------------------------
# Useful function definitions
# -----------------------------------------------------------------------------
def Beta(P):
    """
    beta = P/E
    """
    E = sqrt(P**2+0.511**2)
    beta = P/E

    #print "Beta-> P ={0}, E={1} beta={2}".format(P,E,beta)
    return beta

def SigmaThetaMs(P,L):
    """
    sigma(theta_ms) = 13.6 (Mev)/(beta*P)*Sqrt(LLr)*(1+0-038*log(LLr))
    L in radiation length 
    """
    beta = Beta(P)
    if(beta > 0. and L > 0.):
        tms = (MS0/(P*1.*beta))*sqrt(L*1.)*(1 + 0.038*log(L*1.))
    else:
        tms = 0.;

    #print "SigmaThetaMs->  L={0} P={1} beta={2}, tms={3}".format(L,P,beta,tms)
    return tms

# Read nevts from the file named fn_genbb.
def readGenbbFile(fn_genbb):

    # Open the file.
    fgenbb = open(fn_genbb,'r')
    
    # Read the header (skip 22 lines).
    for nl in range(22): fgenbb.readline()
    
    # Read the events line and following blank line.
    [e1,nevts] = fgenbb.readline().split()
    fgenbb.readline()
    
    nevts = int(nevts)
    print "Reading {0} events from file {1}".format(nevts,fn_genbb)
    
    # Read the particle information for each event.
    p1px = []; p1py = []; p1pz = []
    p2px = []; p2py = []; p2pz = []
    for nevt in range(nevts):
        
        # Read the event number and number of particles (assume 2).
        fgenbb.readline()
        
        # Read the particle momentum information.
        [g1,px1,py1,pz1,t1] = fgenbb.readline().split()
        [g2,px2,py2,pz2,t2] = fgenbb.readline().split()
        p1px.append(px1); p1py.append(py1); p1pz.append(pz1)
        p2px.append(px2); p2py.append(py2); p2pz.append(pz2)

    # Convert to arrays of floats.
    p1px = np.array(p1px).astype(float); p2px = np.array(p2px).astype(float)
    p1py = np.array(p1py).astype(float); p2py = np.array(p2py).astype(float)
    p1pz = np.array(p1pz).astype(float); p2pz = np.array(p2pz).astype(float)
    
    # Return the lists of particle momentum information.        
    return (p1px,p1py,p1pz,p2px,p2py,p2pz)

# A grayscale color map.
# For each color 'red', 'green', 'blue', there is a 3-column table with
# as many rows as necessary.
# col 1 = x, must go from 0 to 1.0
# col 2 = y0
# col 3 = y1
# From matplotlib doc: "For any input value z falling between x[i] and x[i+1], 
#  the output value of a given color will be linearly interpolated between 
#  y1[i] and y0[i+1]"
cdict = {'red': ((0.0, 1.0, 1.0),
                 (0.0, 0.85, 0.85),
                 (1.0, 0.0, 0.0)),
         'green': ((0.0, 1.0, 1.0),
                   (0.0, 0.85, 0.85),
                   (1.0, 0.0, 0.0)),
         'blue': ((0.0, 1.0, 1.0),
                  (0.0, 0.85, 0.85),
                  (1.0, 0.0, 0.0))}
#cdict = {'red': ((0.0, 1.0, 1.0),
#                 (1.0, 0.0, 0.0)),
#         'green': ((0.0, 1.0, 1.0),
#                   (1.0, 0.0, 0.0)),
#         'blue': ((0.0, 1.0, 1.0),
#                  (1.0, 0.0, 0.0))}
tmc_gs_cmap = mpcol.LinearSegmentedColormap('gs_cmap',cdict,256);