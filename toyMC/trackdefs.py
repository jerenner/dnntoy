"""
trackdefs.py

Function definitions and constants for toy MC track generation.

"""
import sys
import numpy as np
import scipy.integrate as integrate
import random as rd
import os
from math import *

# -------------------------------------------------------------------------------------------------------
# Frequently modified parameters:
# -------------------------------------------------------------------------------------------------------

# Output directories
trk_outdir = "/data4/NEXT/users/jrenner/dnn/dnntoy/tracks";   # output directory for tracks

trk_name = "dnn3d_1mm_si";  # name assigned to this run; will be used in naming the output files
trk_bb = True;       # set to true to create "double-beta"-like tracks
num_tracks = 10;  # number of tracks to generate and/or fit

Bfield = 0.0;      # applied magnetic field in Tesla
MS0 = 13.6;        # multiple scattering parameter (should be 13.6)

# Voxelization parameters
vox_ext = 112;     # extent in x,y,z of voxelized track (in +/- directions): full allowed range is 2 * vox_ext
vox_size = 1;      # voxel size
vox_3d = True;    # output 3D voxels or 2D (x-y) projection

# Energy parameters
E_0 = 2.447+0.511; # initial energy in MeV
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
