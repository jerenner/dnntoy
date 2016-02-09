"""
hdf5_combine.py

Combines multiple hdf5 files containing voxelized hits.

Usage:

python hdf5_combine.py <n_runs>

Looks for files named vox_(trk_name)_(#).h5 where (trk_name) is specified
in trackdefs.py, and (#) is calculated starting at 0 and incrementing
by the number of events (num_tracks, also specified in trackdefs.py) until
<n_runs> files have been processed.  The final output file is named
vox_(run_name)_combined.h5

"""
import sys
import h5py
from trackdefs import *

debug = 0

# -----------------------------------------------------------------------------
# Get the arguments, if any.
# -----------------------------------------------------------------------------
args = sys.argv;
if(len(args) < 2):
    print "Usage:\n\n python hdf5_combine.py <n_runs>";
    exit();
trk_nruns = int(args[1]);

fnb_trk = "{0}/{1}".format(trk_outdir,trk_name);

# Open an h5f file to save the results.
h5fc = h5py.File("{0}/vox_{1}_combined.h5".format(fnb_trk,trk_name));

# Process nruns files.
for nrun in range(trk_nruns):
    
    print "Processing run {0}".format(nrun);
        
    # Open the hdf5 file containing the voxels for this run.
    h5fr = h5py.File("{0}/vox_{1}_{2}.h5".format(fnb_trk,trk_name,int(nrun*num_tracks)),'r');
    
    # Save the track as an entry in the hdf5 file.
    for dset in h5fr:
        h5fc.create_dataset(dset,data=h5fr[dset]);
    
    # Close the hdf5 file for this run.
    h5fr.close();

# Close the combined hdf5 file.
h5fc.close();