from   __future__         import print_function
from   keras.optimizers   import SGD, Nadam
from   keras.models       import Sequential, Model
from   keras.layers       import Input, Dense, Activation, Convolution2D, AveragePooling2D, MaxPooling2D, merge, Reshape, Flatten, Dropout
from   keras              import callbacks
from   matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import tables            as tb
import numpy             as np
import copy
import h5py
import sys

# Range limits from extraction of Geant4-simulated events.
NX = 200
NY = 200
NZ = 200

# Projection voxel sizes in mm
vSizeX = 1
vSizeY = 1

# The slice width, in Geant4 voxels
slice_width = 2.

# Range limit in x and y for slices (assuming a square range), in voxels
RNG_LIM = 200

# SiPM plane geometry definition
nsipm = 20             # number of SiPMs in response map (a 10x10 response map covers a 100x100 range)
sipm_pitch = 10.       # distance between SiPMs
sipm_edge_width = 5.   # distance between SiPM and edge of board

# -----------------------------------------------------------------------------------------------------------
# Variables for computing an EL point location
xlen = 2*sipm_edge_width + (nsipm-1)*sipm_pitch       # (mm) side length of rectangle
ylen = 2*sipm_edge_width + (nsipm-1)*sipm_pitch       # (mm) side length of rectangle
wbin = 2.0                                            # (mm) bin width

# Compute the positions of the SiPMs
pos_x = np.ones(nsipm**2)*sipm_edge_width + (np.ones(nsipm*nsipm)*range(nsipm**2) % nsipm)*sipm_pitch
pos_y = np.ones(nsipm**2)*sipm_edge_width + np.floor(np.ones(nsipm*nsipm)*range(nsipm**2) / nsipm)*sipm_pitch

# Number of time bins
n_tbins = 2

# Coefficients from S2 parameterization
M = [1.599, 1.599]
c0 = [7.72708346764e-05, 0.000116782596518]
c1 = [-1.69330613273e-07, 3.05115354927e-06]
c2 = [-1.52173658255e-06, -7.00800605142e-06]
c3 = [-2.4985972302e-07, 6.53907883449e-07]
c4 = [1.12327204397e-07, 8.95230202525e-08]
c5 = [-1.49353264606e-08, -2.27173290582e-08]
c6 = [1.04614146487e-09, 2.00740799864e-09]
c7 = [-4.19111362353e-11, -9.21915945523e-11]
c8 = [9.12129133361e-13, 2.20534216312e-12]
c9 = [-8.40089561697e-15, -2.1795164563e-14]

# Maximum radial extent of parameterization
rmax = 20.

# Return the SiPM response for the specified time bin and radial distance.
def sipm_par(tbin,r):

    # Ensure the time bin value is valid.
    if(tbin < 0 or tbin >= n_tbins):
        print("Invalid time bin in sipm_param: returning 0.0 ...")
        return 0.0

    # Calculate the response based on the parametrization.
    vpar = M[tbin]*(c0[tbin] + c1[tbin]*r + c2[tbin]*r**2 + c3[tbin]*r**3 + 
    c4[tbin]*r**4 + c5[tbin]*r**5 + c6[tbin]*r**6 + c7[tbin]*r**7 + 
    c8[tbin]*r**8 + c9[tbin]*r**9)

    # Zero the response for radii too large.
    if(hasattr(vpar, "__len__")):
        ret = np.zeros(len(vpar)); iret = 0
        for rv,pv in zip(r,vpar):
            if(rv < rmax):
                ret[iret] = pv
            iret += 1
        return ret
    else:
        if(r < rmax):
            return vpar
        return 0.0

# Create slices for the specified event.
#  hfile: the HDF5 files containing the events
#  nevt: the event number to slice
#  zwidth: the slice width in mm
#
#  returns: [energies, slices]
#   where energies is a list of the energies in each slice
#   and slices is a matrix of size [Nslices,NY,NX] containing normalized slices
def slice_evt(hfile,nevt,zwidth):
    
    # Get the event from the file.
    htrk = hfile['trk{0}'.format(nevt)]
    
    # Get the z-range.
    zmin = np.min(htrk[2]); zmax = np.max(htrk[2])
    
    # Create slices of width zwidth beginning fom zmin.
    nslices = int(np.ceil((zmax - zmin)/zwidth))
    #print "{0} slices for event {1}".format(nslices,nevt)
    
    slices = np.zeros([nslices,NY,NX])
    energies = np.zeros(nslices)
    for x,y,z,e in zip(htrk[0],htrk[1],htrk[2],htrk[3]):
        
        # Add the energy at (x,y,z) to the (x,y) value of the correct slice.
        islice = int((z - zmin)/zwidth)
        if(islice == nslices): islice -= 1
        slices[islice][y][x] += e
        energies[islice] += e
    
    # Normalize the slices.
    for s in range(nslices):
        slices[s] /= energies[s]
        
    # Return the list of slices and energies.
    return [energies, slices]


xrng = []; yrng = []   # x- and y-ranges
nspevt = []            # number of slices per event
slices_x = []; slices_y = []; slices_e = []   # slice arrays
for ee in range(Ntrks):

    en_evt = sum(en)    

    if(ee % int(Ntrks/100) == 0):
        print("Slicing event {0}".format(ee))
        
    # Slice the event.
    en,sl = slice_evt(h5f,ee,slice_width)
    nslices = len(en)
    nspevt.append(nslices)
    
    # Get information about each slice.
    for ss in range(nslices):
        
        # Don't include 0-energy slices.
        if(en[ss] < 1.0e-6):
            continue
        
        # Get lists of the nonzero x,y,z indices and E values.
        cslice = sl[ss]
        nzy,nzx = np.nonzero(cslice)
        nze = cslice[np.nonzero(cslice)]
        
        # Extract several quantities of interest.
        xmin = np.min(nzx); xmax = np.max(nzx)
        ymin = np.min(nzy); ymax = np.max(nzy)
        xrng.append(xmax - xmin + 1)
        yrng.append(ymax - ymin + 1)
        
        # Save the slice if within range.
        if((xmax - xmin) >= RNG_LIM-1 or (ymax - ymin) >= RNG_LIM-1):
            print("Range of {0} for event {1} slice {2}, energy {3}; slice not included".format(xmax-xmin,ee,ss,en[ss]))
        else:
            
            # Center the slices about RNG_LIM/2.
            #x0 = int((xmin + xmax)/2. - RNG_LIM/2.)
            #y0 = int((ymin + ymax)/2. - RNG_LIM/2.)
            #nzx -= x0; nzy -= y0
            
            # Create the slice array.
            snum = len(slices_x)
            slices_x.append(nzx); slices_y.append(nzy); slices_e.append(nze)
            carr = np.array([nzx, nzy, nze])
            
            # Create the corresponding SiPM map.
            sipm_map = np.zeros(nsipm*nsipm)
            for xpt,ypt,ept in zip(nzx,nzy,nze):

                # Compute the distances and probabilities.  Add the probabilities to the sipm map.
                rr = np.array([np.sqrt((xi - xpt)**2 + (yi - ypt)**2) for xi,yi in zip(pos_x,pos_y)])
                probs = 0.5*(sipm_par(0, rr) + sipm_par(1, rr))
                sipm_map += probs*ept

            # Multiply the SiPM map by a factor proportional to the slice energy.
            sipm_map *= en[ss]/en_evt

        # Normalize the probability map, and set sigma = 1.
        #sipm_map -= np.mean(sipm_map)
        #sipm_map /= np.std(sipm_map)
            
        # Save the slice and the SiPM map to an HDF5 file.
        h5slices.create_dataset("slice{0}".format(snum),data=carr)
        h5slices.create_dataset("sipm{0}".format(snum),data=sipm_map)
            
h5slices.close()
