"""
dnnplot.py

Plots tracks and results from DNN analyses.

1.  plot individual tracks:

    python dnnplot.py tracks <evt_start> <evt_end>

    -- Example: python dnnplot.py tracks 0 5 
                (Plots tracks 0 through 4 in the vox_dnn3d_NEXT100_Paolina222_v10x10x10_r200x200x200_si.h5 dataset)

2.  plot run summary:

    python dnnplot.py summary

    -- Example: python dnnplot.py summary
                (Plots summary for run testrun)

3.  plot signal vs. background curve for given epoch:

    python dnnplot.py svsb <epoch>

    -- Example: python dnnplot.py svsb 5
                (Plots signal vs. background curve for epoch 5 of run testrun)

"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from math import *
from dnninputs import *

# -----------------------------------------------------------------------------
# Get the arguments
# -----------------------------------------------------------------------------
usage_str = "Usage:\n\n python dnnplot.py <type> (<start>) (<end>)"
args = sys.argv

# Must have at least 2 arguments.
if(len(args) < 2):
    print usage_str
    exit();

# Get the run name and type of plot.
ptype = args[1]

evt_start = -1; evt_end = -1
epoch = -1
# If we are plotting tracks, get the start and end events.
if(ptype == "tracks"):
    evt_start = int(args[2])
    evt_end = int(args[3])
# If we are plotting signal vs. background, get the epoch.
elif(ptype == "svsb"):
    epoch = int(args[2])
# Otherwise we should be plotting the summary.
elif(ptype != "summary"):
    print usage_str
    exit()

# -----------------------------------------------------------------------------
# File names and directories
# -----------------------------------------------------------------------------
fn_summary = "{0}/{1}/acc/accuracy_{2}.dat".format(rdir,rname,rname)
fn_svsb = "{0}/{1}/acc/prob_{2}_test_ep{3}.dat".format(rdir,rname,rname,epoch)

if(not os.path.isdir("{0}/{1}/plt".format(rdir,rname))): os.mkdir("{0}/{1}/plt".format(rdir,rname))
if(not os.path.isdir("{0}/plt".format(datdir))): os.mkdir("{0}/plt".format(datdir))
if(not os.path.isdir("{0}/plt/{1}".format(datdir,rname))): os.mkdir("{0}/plt/{1}".format(datdir,rname))

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

# Summary plot
if(ptype == "summary"):

    print "Plotting summary..."

    # Read in the results.
    accmat = np.loadtxt(fn_summary)
    acc_trs = accmat[:,1]*100.
    acc_trb = accmat[:,2]*100.
    acc_vls = accmat[:,5]*100.
    acc_vlb = accmat[:,6]*100.
    acc_itr = [] 
    for iit in range(len(acc_trs)): acc_itr.append(iit)

    # Plot the results.
    fig = plt.figure(1);
    fig.set_figheight(5.0);
    fig.set_figwidth(15.0);

    ax1 = fig.add_subplot(121);
    ax1.plot(acc_itr, acc_trs, '-', color='blue', lw=1, label='Training (si)')
    ax1.plot(acc_itr, acc_trb, '-', color='green', lw=1, label='Training (bg)')
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("accuracy")
    ax1.set_title("")
    ax1.set_ylim([0, 100]);
    #ax1.set_xscale('log')

    lnd = plt.legend(loc=4,frameon=False,handletextpad=0)

    ax2 = fig.add_subplot(122);
    ax2.plot(acc_itr, acc_vls, '-', color='blue', lw=1, label='Validation (si)')
    ax2.plot(acc_itr, acc_vlb, '-', color='green', lw=1, label='Validation (bg)')
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.set_title("")
    ax2.set_ylim([0, 100]);
    #ax2.set_xscale('log')

    lnd = plt.legend(loc=4,frameon=False,handletextpad=0)

    # Show and/or print the plot.
    fn_plt = "{0}/{1}/plt/{2}_summary.png".format(rdir,rname,rname)
    plt.savefig(fn_plt, bbox_inches='tight')
    if(plt_show):
        plt.show()
    plt.close()

# Signal vs. background curve
if(ptype == "svsb"):

    accmat = np.loadtxt(fn_svsb)
    acc_etype = accmat[:,0]
    acc_psi = accmat[:,1]
    acc_pbg = accmat[:,2]

    # Plot the results.
    fig = plt.figure(1);
    fig.set_figheight(5.0);
    fig.set_figwidth(15.0);

    ax1.plot(acc_ep, acc_trb, '-.', color='red', lw=1)
    ax1.set_xlabel("Background rejection")
    ax1.set_ylabel("Signal efficiency")
    ax1.set_title("")

    #lnd = plt.legend(loc=4,frameon=False,handletextpad=0)

    # Show and/or print the plot.
    fn_plt = "{0}/{1}/plt/{2}_svsb_ep{3}.png".format(rdir,rname,rname,epoch)
    plt.savefig(fn_plt, bbox_inches='tight')
    if(plt_show):
        plt.show()
    plt.close()

# Tracks
if(ptype == "tracks"):

    print "Plotting tracks {0} to {1}...".format(evt_start,evt_end)

    # 

