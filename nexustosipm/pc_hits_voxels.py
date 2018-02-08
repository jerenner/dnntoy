import os
import random
import sys
import textwrap
import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy  as np
import tables
import h5py

import invisible_cities.core.core_functions as coref
import invisible_cities.core.fit_functions as fitf
import invisible_cities.reco.dst_functions  as dstf
import invisible_cities.reco.paolina_functions as plf

from   collections                             import namedtuple
from   invisible_cities.io.hits_io             import load_hits
from   invisible_cities.io.hits_io             import load_hits_skipping_NN
from   invisible_cities.evm.event_model        import Hit, Cluster, HitCollection
from   invisible_cities.types.ic_types         import xy
from   invisible_cities.types.ic_types         import NN
from   invisible_cities.core.system_of_units_c import units
from   matplotlib.patches                      import Ellipse
from   mpl_toolkits.mplot3d                    import Axes3D

# Reconstructed hits
class Reco(tables.IsDescription):
    """
    Stores reconstructed hits 
    """
    event_indx = tables.Int32Col(pos=0)
    nof_hits = tables.Int16Col(pos=1)
    hit_indx = tables.Int16Col(pos=2)
    hit_position = tables.Float32Col(shape=3, pos=3)
    hit_energy = tables.Float32Col(pos=4)
    hit_geocorr = tables.Float32Col(pos=5)
    hit_ltcorr = tables.Float32Col(pos=6)

##################################################################################
## PARAMETERS
vox_size = np.array([5,5,5]).astype('int')    # voxel size
blob_radius = 21.                    # blob radius in mm
rng = 400                            # total range of event in mm
vdim = (rng/vox_size).astype('int')                  # number of voxels along each dimension

###################################################################################
## ARGUMENTS
def get_parser(args=None):
    parser = argparse.ArgumentParser(description='Script to produce HDF5 files')
    parser.add_argument('-i','--hitsfile',
                        action='store',
                        help='hits file',
                        required='True')
    parser.add_argument('-c','--cfile',
                        action='store',
                        help='correction table file',
                        required='True')
    parser.add_argument('-o','--ofile',
                        action='store',
                        help='output file',
                        required='True')
    parser.add_argument('-t','--tlife',
                        action='store',
                        help='lifetime',
                        required='True')
    parser.add_argument('-f','--ftlife',
                        action='store',
                        help='lifetime scale factor',
                        required='False')
    return parser

args = get_parser().parse_args()
opts = vars(args) # dict

hits_file = args.hitsfile
corr_file = args.cfile
evt_file  = args.ofile
tlife     = float(args.tlife)
ftlife = 1.0
if(args.ftlife):
    ftlife     = float(args.ftlife)
    print("-- Read lifetime scale factor = {}".format(ftlife))

##################################################################################
## PRINT CONFIGURATION
print("** Running correction + Paolina step for the following configuration:")
print("-- hits file = {}".format(hits_file))
print("-- correction table = {}".format(corr_file))
print("-- output file = {}".format(evt_file))
print("-- voxel size = {0}x{1}x{2} cubic mm".format(vox_size[0],vox_size[1],vox_size[2]))

#################################################################################
## FUNCTION DEFINITIONS
def gaussexpo(x, amp, mu, sigma, const, mean, x0):

    if sigma <= 0.:
        return np.inf
    
    return amp/(2*np.pi)**.5/sigma * np.exp(-0.5*(x-mu)**2./sigma**2.) + const * np.exp((x-x0)/mean)

# code from Gonzalo
def reso(values):
    _, mu, sigma = values
    r = 235. * sigma/mu
    return r, r * (1592/2458)**0.5

def reso_errors(values,errors):
    _, mu, sigma = values
    _, err_mu, err_sigma = errors
    r = 235. * sigma/mu
    err_r = r*np.sqrt((err_sigma/sigma)**2 + (err_mu/mu)**2)
    return err_r, err_r * (1592/2458)**0.5

def gausstext(values):
    return textwrap.dedent("""
        $\mu$ = {:.1f}
        $\sigma$ = {:.2f}
        R = {:.3}%
        Rbb = {:.3}%""".format(*values[1:], *reso(values)))

def gaussexpotext(values):
    return textwrap.dedent("""
        $\mu$ = {:.1f}
        $\sigma$ = {:.2f}
        $\\bar x$ = {:.2f}
        R = {:.3}%
        Rbb = {:.3}%""".format(*values[1:3], values[4], *reso(values[0:3])))

###############################################################################################
## CORRECTIONS TABLE
XYcorr = dstf.load_xy_corrections(corr_file, group = "XYcorrections", node = f"GeometryE_5.0mm", norm_strategy = "const", norm_opts = {"value": 41.5})
LTcorr = dstf.load_lifetime_xy_corrections(corr_file, group = "XYcorrections", node  =  "Lifetime")
XYQcorr = dstf.load_xy_corrections(corr_file, group = "XYcorrections", node = f"GeometryQ_5.0mm", norm_strategy = "index", norm_opts = {"index": (40, 40)})
LTQcorr = dstf.load_lifetime_xy_corrections(corr_file, group = "XYcorrections", node  =  "QLifetime")

# Read in the output of Penthesilea.
def merge_NN_hits(hits_all,hits_nonNN):

    # Iterate through the nonNN dictionary and update the energies including the NN hits from the "all" dictionary.
    for (evt,hc) in hits_nonNN.items():

        # Get the corresponding collection of all hits.
        hc_all = hits_all[evt]
            
        # Add energy from all NN hits to hits in closest slice.
        for h1 in hc_all.hits:

            if(h1.Q == NN):

                # Find the hits to which the energy will be added.
                zdist_min = -1
                h_add = []
                for h2 in hc.hits:
                    zdist = np.abs(h1.Z - h2.Z)
                    if(zdist_min < 0 or zdist < zdist_min):
                        zdist_min = zdist
                        h_add = []
                        h_add.append(h2)
                    elif(zdist == zdist_min):
                        h_add.append(h2)

                # Add the energy.
                hadd_etot = sum([ha.E for ha in h_add])
                for ha in h_add:
                    ha.energy += h1.E*(ha.E/hadd_etot)
                    
        # Check the sum of the energy.
        #e1 = sum([hh.E for hh in hc_all.hits])
        #e2 = sum([hh.E for hh in hc.hits])
        #if(abs(e1 - e2) > 0.001):
        #    print("ERROR")

#########################################################################################################
## HITS
if(os.path.isfile(hits_file)):
    hits_all = load_hits(hits_file)
    hits = load_hits_skipping_NN(hits_file)

    # Modifies the list of non-NN hits.
    merge_NN_hits(hits_all,hits)
else:
    print("ERROR: hits not found.")
    exit()

# Open the Pytables file to contain the hits information.
h5f = tables.open_file(evt_file+'_hits.h5', "w", filters=tables.Filters(complib="blosc", complevel=9))
group_Reco = h5f.create_group(h5f.root, "Reco")
Reco_table = h5f.create_table(group_Reco, "Reco", Reco, "Reco", tables.Filters(0))

fvox = h5py.File(evt_file+'_voxels.h5')

# Create the corrected hit collections (summed over all runs) for 
#  fully corrected (c), geometry-only corrected (g), tau-only corrected (t), geometry+global tau corrected (gtglobal), and uncorrected (u) events.
hitc_uevt = []; hitc_cevt = []; hitc_gevt = []; hitc_tevt = []; hitc_gtglobalevt = []
A_evtnum = []
for ee,hc in hits.items():
    hc_ucorr = []; hc_corr = []; hc_gcorr = []; hc_tcorr = []; hc_gtglobalcorr = []
    ##   AP ---------
    geo_corr = []; lt_corr = []; hit_x = []; hit_y = []; hit_z=[]; hit_E = []
    ##  AP end ---------

    # Initial hit correction for charge variation.
#    hcollection = hc.hits
#    hn = 0
#    while(hn < len(hcollection)):
# 
#        # Create a sub-collection with all hits of the same z. 
#        hqcoll = []
#        hcurrent = hcollection[hn]
#        hqcoll.append(hcurrent)
#        hn += 1
#        while(hn < len(hcollection) and hcollection[hn].Z == hcurrent.Z):
#            hqcoll.append(hcollection[hn])
#            hn += 1
#        
#        # Perform an additional correction if necessary.
#        if(len(hqcoll) > 1):
#            #print("Correcting {} hits".format(len(hqcoll)))
#            Qtot = sum([hh.Q for hh in hqcoll])
#            Qptot = sum([hh.Q*XYQcorr(hh.X,hh.Y).value*LTQcorr(hh.Z,hh.X,hh.Y).value for hh in hqcoll])
#
#            # Only correct if we have nonzero Q'.
#            if(Qptot > 0):
#                for hh in hqcoll:
#                    hh.energy = hh.E*(Qtot/Qptot)*XYQcorr(hh.X,hh.Y).value*LTQcorr(hh.Z,hh.X,hh.Y).value
#        #else:
#        #    print("Not correcting for {} hits".format(len(hqcoll)))

    for hh in hc.hits:

        hecorr  = hh.E*XYcorr(hh.X,hh.Y).value*LTcorr(hh.Z,hh.X,hh.Y).value**(ftlife) #/np.exp(-hh.Z/tlife)
        hegcorr = hh.E*XYcorr(hh.X,hh.Y).value
        hetcorr = hh.E*LTcorr(hh.Z,hh.X,hh.Y).value**(ftlife) #/np.exp(-hh.Z/tlife)
        hegtglobalcorr = hh.E*XYcorr(hh.X,hh.Y).value/np.exp(-hh.Z/tlife)
            
        hucorr = Hit(0,Cluster(0, xy(hh.X,hh.Y), xy(0,0), 0),hh.Z,hh.E)
        hcorr  = Hit(0,Cluster(0, xy(hh.X,hh.Y), xy(0,0), 0),hh.Z,hecorr)
        hgcorr = Hit(0,Cluster(0, xy(hh.X,hh.Y), xy(0,0), 0),hh.Z,hegcorr)
        htcorr = Hit(0,Cluster(0, xy(hh.X,hh.Y), xy(0,0), 0),hh.Z,hetcorr)
        hgtglobalcorr = Hit(0,Cluster(0, xy(hh.X,hh.Y), xy(0,0), 0),hh.Z,hegtglobalcorr)
            
        hc_ucorr.append(hucorr); hc_corr.append(hcorr); hc_gcorr.append(hgcorr); hc_tcorr.append(htcorr); hc_gtglobalcorr.append(hgtglobalcorr)

    # Only save events with >= 2 hits.
    if(len(hc_corr) >= 2):
        hitc_uevt.append(hc_ucorr)
        hitc_cevt.append(hc_corr)
        hitc_gevt.append(hc_gcorr)
        hitc_tevt.append(hc_tcorr)
        hitc_gtglobalevt.append(hc_gtglobalcorr)

        # Save the event number for later use.
        A_evtnum.append(ee)

        # Save the hits.
        reco_ = Reco_table.row
        vtx = np.zeros(3)
        for ih,hh in enumerate(hc_corr):
            reco_["event_indx"] = ee
            reco_["nof_hits"] = len(hc_corr)
            reco_["hit_indx"] = ih

            vtx[0] = hh.X
            vtx[1] = hh.Y
            vtx[2] = hh.Z
            reco_["hit_position"] = vtx
            reco_["hit_energy"] = hh.E
            reco_["hit_geocorr"] = XYcorr(hh.X,hh.Y).value
            reco_["hit_ltcorr"] = LTcorr(hh.Z,hh.X,hh.Y).value**(ftlife)
            
            reco_.append()

A_evtnum = np.array(A_evtnum)

# Run Paolina for many events: note we now assume that all events have >= 2 hits.
A_eblob1 = []; A_eblob2 = []; A_emtrk = []; A_ntrks = []; A_lmtrk = []; A_nvox = []
for nevt in range(len(hitc_cevt)):
    
    hitc = hitc_cevt[nevt]
    print("Track {0} with {1} hits...".format(nevt,len(hitc)))

    # Make the tracks.
    voxels = plf.voxelize_hits(hitc,vox_size)
    trks = plf.make_track_graphs(voxels)
    l_etrks_all = []
    for t in trks:
        if(len(t.nodes()) < 1):
            etrk = 0
        else:
            etrk = sum([vox.E for vox in t.nodes()])
        l_etrks_all.append(etrk)
    
    Eblob1 = -1; Eblob2 = -1; etmax = -1; ltrk = -1
    if(len(l_etrks_all) > 0):
        
        # "max" track is the one with the most energy.
        itmax = np.argmax(l_etrks_all) 
        etmax = sum([vox.E for vox in trks[itmax].nodes()])

        # Construct the blobs.
        eblobs = plf.blob_energies(trks[itmax],blob_radius)
        iter_eblobs = iter(eblobs)
        Eblob1, Eblob2 = next(iter_eblobs),next(iter_eblobs)

        # Ensure blob1 always has higher energy.
        if(Eblob2 > Eblob1):
            eswap = Eblob1
            Eblob1 = Eblob2
            Eblob2 = eswap

        # Get the extremes.
        distances = plf.shortest_paths(trks[itmax])
        a,b = plf.find_extrema(trks[itmax]) #(distances)
        ltrk = distances[a][b]
        print("Found {0} tracks of {1}; max containing {2} voxels; total of {3} voxels, distance = {4}".format(len(trks),len(hitc_cevt),len(trks[itmax]),len(voxels),distances[a][b]))

    # Center the voxels.
    xmin = np.min([vox.X for vox in voxels]); xmax = np.max([vox.X for vox in voxels])
    ymin = np.min([vox.Y for vox in voxels]); ymax = np.max([vox.Y for vox in voxels])
    zmin = np.min([vox.Z for vox in voxels]); zmax = np.max([vox.Z for vox in voxels])
    x0 = ((xmin + xmax) / 2. - (rng / 2.))
    y0 = ((ymin + ymax) / 2. - (rng / 2.))
    z0 = ((zmin + zmax) / 2. - (rng / 2.))

    # Create the voxel array.
    varr = np.zeros([vdim[0],vdim[1],vdim[2]])

    # Iterate through the voxels, applying offsets.
    valid_voxels = True
    for vv in voxels:
        ivox = int((vv.X - x0)/vox_size[0])
        jvox = int((vv.Y - y0)/vox_size[1])
        kvox = int((vv.Z - z0)/vox_size[2])
        evox = vv.E
        if(ivox < 0 or ivox >= vdim[0] or jvox < 0 or jvox >= vdim[1] or kvox < 0 or kvox >= vdim[2]):
            print("WARNING: event {} out of range".format(nevt))
            valid_voxels = False
        else:
            if(varr[ivox][jvox][kvox] > 0.):
                print("WARNING: duplicate voxel in event {}".format(nevt))
            varr[ivox][jvox][kvox] += vv.E

    # Include the event in the voxels output if it was valid.
    if(valid_voxels):

        # Get lists of the nonzero x,y,z indices and E values.
        nzx,nzy,nzz = np.nonzero(varr)
        nze = varr[np.nonzero(varr)]

        # Combine into one single 4 x N array, where N is the number of nonzero elements.
        carr = np.array([nzx, nzy, nzz, nze])

        # Add the track.
        fvox.create_dataset("trk{0}".format(A_evtnum[nevt]),data=carr)

    # Add to the distributions.
    A_eblob1.append(Eblob1)
    A_eblob2.append(Eblob2)
    A_emtrk.append(etmax)
    A_ntrks.append(len(trks))
    A_lmtrk.append(ltrk)
    A_nvox.append(len(voxels))

# Convert to numpy arrays.
A_eblob1 = np.array(A_eblob1)
A_eblob2 = np.array(A_eblob2)
A_emtrk  = np.array(A_emtrk)
A_ntrks   = np.array(A_ntrks)
A_lmtrk  = np.array(A_lmtrk)
A_nvox   = np.array(A_nvox)

# Compute key quantities.
A_Ec = []; A_Ec_avg = []; A_Ec_tau = []; A_Ec_geo = []; A_Ec_gtglobal = []; A_E0 = []
A_xavg = []; A_yavg = []; A_zavg = []; A_ravg = []
A_xmin = []; A_ymin = []; A_zmin = []
A_xmax = []; A_ymax = []; A_zmax = []
A_rmin = []; A_rmax = []
for ee in range(len(hitc_cevt)):
    
    # Compute the corrected energy and average coordinates.
    evt_E = sum([hh.E for hh in hitc_cevt[ee]])
    evt_X = sum([hh.X*hh.E for hh in hitc_cevt[ee]])
    evt_Y = sum([hh.Y*hh.E for hh in hitc_cevt[ee]])
    evt_Z = sum([hh.Z*hh.E for hh in hitc_cevt[ee]])
    if(evt_E > 0):
        evt_X /= evt_E
        evt_Y /= evt_E
        evt_Z /= evt_E
    evt_R = np.sqrt(evt_X**2 + evt_Y**2)

    # Get the minimum and maximum coordinate values.
    evt_xmin = min([hh.X for hh in hitc_cevt[ee]])
    evt_ymin = min([hh.Y for hh in hitc_cevt[ee]])
    evt_zmin = min([hh.Z for hh in hitc_cevt[ee]])
    evt_xmax = max([hh.X for hh in hitc_cevt[ee]])
    evt_ymax = max([hh.Y for hh in hitc_cevt[ee]])
    evt_zmax = max([hh.Z for hh in hitc_cevt[ee]])
    evt_rmin = min([np.sqrt(hh.X**2 + hh.Y**2) for hh in hitc_cevt[ee]])
    evt_rmax = max([np.sqrt(hh.X**2 + hh.Y**2) for hh in hitc_cevt[ee]])

    # Compute the energy with other corrections.
    evt_E_uncorr = sum([hh.E for hh in hitc_uevt[ee]])
    evt_E_cgeo = sum([hh.E for hh in hitc_gevt[ee]])
    evt_E_ctau = sum([hh.E for hh in hitc_tevt[ee]])
    evt_E_cgtglobal = sum([hh.E for hh in hitc_gtglobalevt[ee]])
    
    # Add to distributions.
    A_Ec.append(evt_E)
    A_Ec_avg.append(evt_E_uncorr*XYcorr(evt_X,evt_Y).value*LTcorr(evt_Z,evt_X,evt_Y).value**(ftlife)) #/np.exp(-evt_Z/tlife))
    A_Ec_tau.append(evt_E_ctau)
    A_Ec_geo.append(evt_E_cgeo)
    A_Ec_gtglobal.append(evt_E_cgtglobal)
    A_E0.append(evt_E_uncorr)
    A_xavg.append(evt_X)
    A_yavg.append(evt_Y)
    A_zavg.append(evt_Z)
    A_ravg.append(evt_R)
    A_xmin.append(evt_xmin)
    A_ymin.append(evt_ymin)
    A_zmin.append(evt_zmin)
    A_xmax.append(evt_xmax)
    A_ymax.append(evt_ymax)
    A_zmax.append(evt_zmax)
    A_rmin.append(evt_rmin)
    A_rmax.append(evt_rmax)

# Convert to numpy arrays.
A_Ec = np.array(A_Ec)
A_Ec_avg = np.array(A_Ec_avg)
A_Ec_tau = np.array(A_Ec_tau)
A_Ec_geo = np.array(A_Ec_geo)
A_Ec_gtglobal = np.array(A_Ec_gtglobal)
A_E0 = np.array(A_E0)
A_xavg = np.array(A_xavg)
A_yavg = np.array(A_yavg)
A_zavg = np.array(A_zavg)
A_ravg = np.array(A_ravg)
A_xmin = np.array(A_xmin)
A_ymin = np.array(A_ymin)
A_zmin = np.array(A_zmin)
A_xmax = np.array(A_xmax)
A_ymax = np.array(A_ymax)
A_zmax = np.array(A_zmax)
A_rmin = np.array(A_rmin)
A_rmax = np.array(A_rmax)

print("Events after Paolina analysis: {0}".format(len(A_eblob1)))
print("Events in key quantities: {0}".format(len(A_Ec)))

# Save the arrays for future access.
np.savez(evt_file,
        A_evtnum=A_evtnum, 
        A_eblob1=A_eblob1, A_eblob2=A_eblob2, A_emtrk=A_emtrk, A_lmtrk=A_lmtrk, A_ntrks=A_ntrks, A_nvox=A_nvox,
        A_Ec=A_Ec, A_Ec_avg=A_Ec_avg, A_Ec_tau=A_Ec_tau, A_Ec_geo=A_Ec_geo, A_Ec_gtglobal=A_Ec_gtglobal, A_E0=A_E0, 
        A_xavg=A_xavg, A_yavg=A_yavg, A_zavg=A_zavg, A_ravg=A_ravg,
        A_xmin=A_xmin, A_ymin=A_ymin, A_zmin=A_zmin,
        A_xmax=A_xmax, A_ymax=A_ymax, A_zmax=A_zmax,
        A_rmin=A_rmin, A_rmax=A_rmax)

# Save the pytable containing the hits information.
Reco_table.flush()
h5f.close()
fvox.close()
