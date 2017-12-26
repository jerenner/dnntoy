from   __future__         import print_function
from   matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import tables            as tb
import numpy             as np
import copy
import h5py
import sys
import argparse

def get_parser(args=None):
    parser = argparse.ArgumentParser(description='Script to produce HDF5 files')
    parser.add_argument('-j','--jobs',
                        action='store',
                        help='jobs',
                        required='True')
    parser.add_argument('-i','--ijob',
                        action='store',
                        help='job number',
                        required='True')
    parser.add_argument('-n','--nevts',
                        action='store',
                        help='number of events',
                        required='True')
    parser.add_argument('-f','--ifile',
                        action='store',
                        help='input file',
                        required='True')

    return parser


#get options
args = get_parser().parse_args()
opts = vars(args) # dict

jobs = int(args.jobs)
ijob = int(args.ijob)
nevts = int(args.nevts)
infname = args.ifile
outfname = "{}_out_{}.h5".format(infname.split('.')[0],ijob)

# Determine the first and last event for this job.
evtsperjob = int(nevts/jobs)
nievt = ijob*evtsperjob
nfevt = (ijob+1)*evtsperjob
if(nfevt > nevts or (ijob + 1 == jobs)): nfevt = nevts

ulight_frac = 3.0e-7     # scale factor for uniform reflected light: energy E emitted from a single point
                         #   will give rise to a uniform illumination of the SiPM plane  in addition to 
                         #   its usual light cone.  The amount of illumination will be a uniform value with
                         #   with min 0 and max E*sipm_par(0,0)*ulight_frac.

E_to_Q = 0.8e6             # energy to Q (pes) conversion factor

## Initial setup configuration.
vox_ext = 500
vox_sizeX = 2
vox_sizeY = 2

# Slices: number of x and y values.
NX = int(vox_ext/vox_sizeX)
NY = int(vox_ext/vox_sizeY)

# The slice width, in Geant4 voxels
slice_width = 5

# Range limit in x and y for slices (assuming a square range), in voxels
RNG_LIM = vox_ext

# SiPM plane geometry definition
nsipm = 48             # number of SiPMs in one dimension of the response map (a 10x10 response map covers a 100x100 range)
sipm_pitch = 10.       # distance between SiPMs
sipm_edge_width = 5.   # distance between SiPM and edge of board

# -----------------------------------------------------------------------------------------------------------
# Variables for computing an EL point location
xlen = 2*sipm_edge_width + (nsipm-1)*sipm_pitch       # (mm) side length of rectangle
ylen = 2*sipm_edge_width + (nsipm-1)*sipm_pitch       # (mm) side length of rectangle
wbin = 3.0                                            # (mm) bin width

# Compute the positions of the SiPMs
pos_x = np.ones(nsipm**2)*sipm_edge_width + (np.ones(nsipm*nsipm)*range(nsipm**2) % nsipm)*sipm_pitch
pos_y = np.ones(nsipm**2)*sipm_edge_width + np.floor(np.ones(nsipm*nsipm)*range(nsipm**2) / nsipm)*sipm_pitch

# -----------------------------------------------------------------------------------------------------------
# SiPM parameterization

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

# Parameters for theoretical light cone function.
A = 1.
d = 5.
ze = 4.
def sipm_lcone(r):
    v = (A/(4*np.pi*d*np.sqrt(r**2 + ze**2)))*(1 - np.sqrt((r**2 + ze**2)/(r**2 + (ze+d)**2)))
    return v

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
    print("{} slices for event {} with zmax = {} and zmin = {}".format(nslices,nevt,zmax,zmin))
    
    slices = np.zeros([nslices,NY,NX])
    energies = np.zeros(nslices)
    for x,y,z,e in zip(htrk[0].astype('int'),htrk[1].astype('int'),htrk[2].astype('int'),htrk[3]):
        
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

# Open the file containing the voxels.
voxfile = h5py.File(infname,'r')

# Open a file to which the events will be saved.
h5maps = tb.open_file(outfname, 'w')
filters = tb.Filters(complib='blosc', complevel=9, shuffle=False)
atom_m = tb.Atom.from_dtype(np.dtype('float32'))
maparray = h5maps.create_earray(h5maps.root, 'maps', atom_m, (0, nsipm, nsipm, nsipm), filters=filters)
atom_e = tb.Atom.from_dtype(np.dtype('float32'))
earray = h5maps.create_earray(h5maps.root, 'energies', atom_e, (0, nsipm), filters=filters)

# Open the file containing the SiPMs to be nullified by default.
fzsipms = np.load("/home/jrenner/jerenner/dnntoy/nexustosipm/omit_sipms.npz")
zsipms = fzsipms['zsipms']

xrng = []; yrng = []   # x- and y-ranges
nspevt = []            # number of slices per event
slices_x = []; slices_y = []; slices_e = []   # slice arrays
for ee in range(nievt,nfevt,1):

    if(ee % int((nfevt-nievt)/100 + 1) == 0):
        print("Slicing event {0}".format(ee))
        
    # Slice the event.
    en,sl = slice_evt(voxfile,ee,slice_width)
    nslices = len(en)
    nspevt.append(nslices)

    # Calculate the total event energy.
    en_evt = sum(en)
    
    # Create a 2D sipm map from each slice and add it to the final 3D matrix.
    valid_evt = True
    sipm_matrix = np.zeros([nsipm,nsipm,nsipm]).astype('float32')
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
            print("Range of {0} for event {1} slice {2}, energy {3}; event not included".format(xmax-xmin,ee,ss,en[ss]))
            valid_evt = False
        elif(valid_evt):
            
            # Create the corresponding SiPM map.
            sipm_map = np.random.poisson(0.08,size=nsipm*nsipm).astype('float32') #np.zeros(nsipm*nsipm).astype('float32')
            umean = en[ss]*E_to_Q*ulight_frac
            sipm_map += np.random.normal(umean,np.sqrt(umean),size=nsipm*nsipm).astype('float32')
            for xpt,ypt,ept in zip(nzx,nzy,nze):

                # Compute the distances and probabilities.  Add the probabilities to the sipm map.
                rr = np.array([np.sqrt((xi - xpt*vox_sizeX)**2 + (yi - ypt*vox_sizeY)**2) for xi,yi in zip(pos_x,pos_y)])
                probs = sipm_lcone(rr) #0.5*(sipm_par(0, rr) + sipm_par(1, rr))
                sipm_map += probs*ept*E_to_Q

            # Apply the 1-pe threshold.
            sipm_map[sipm_map < 1] = 0.

            # Zero the disabled SiPMs.
            sipm_map[zsipms] = 0.

            # Normalize the SiPM map.
            #sipm_map /= np.sum(sipm_map)

            # Multiply the SiPM map by a factor proportional to the slice energy.
            #sipm_map *= en[ss]/en_evt
            #print("slice {} with energy {}".format(ss,en[ss]))

            # Reshape the map and add it to the list of maps.
            sipm_map = sipm_map.reshape(nsipm,nsipm)
            sipm_matrix[:,:,ss] += sipm_map

    # Save the SiPM map to an HDF5 file.
    if(valid_evt): 
        envector = np.zeros(nsipm)
        envector[0:len(en)] += en
        maparray.append([sipm_matrix])
        earray.append([envector])

voxfile.close()
h5maps.close()
