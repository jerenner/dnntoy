"""
dnn2d_accplot.py

Plots training and validation accuracy throughout DNN training.

"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from math import *

# -----------------------------------------------------------------------------
# Get the arguments
# -----------------------------------------------------------------------------
args = sys.argv;
if(len(args) < 1):
    print "Usage:\n\n python dnn2d_accplot.py <accuracy_file>";
    exit();
fn_acc = args[1]

# Read in the results.
accmat = np.loadtxt("{0}.dat".format(fn_acc))
acc_tr = accmat[:,0]
acc_tns = accmat[:,1]
acc_tnb = accmat[:,2]
acc_itr = [] 
for iit in range(len(acc_tr)): acc_itr.append(iit)

# Plot the results.
fig = plt.figure(1);
fig.set_figheight(5.0);
fig.set_figwidth(15.0);

ax1 = fig.add_subplot(121);
ax1.plot(acc_itr, acc_tr, '-', color='red', lw=1, label='Training')
ax1.set_xlabel("iteration")
ax1.set_ylabel("accuracy")
ax1.set_title("")
#ax1.set_xscale('log')

lnd = plt.legend(loc=4,frameon=False,handletextpad=0)

ax2 = fig.add_subplot(122);
ax2.plot(acc_itr, acc_tns, '-', color='blue', lw=1, label='Validation (si)')
ax2.plot(acc_itr, acc_tnb, '-', color='green', lw=1, label='Validation (bg)')
ax2.set_xlabel("iteration")
ax2.set_ylabel("accuracy")
ax2.set_title("")
#ax2.set_xscale('log')

lnd = plt.legend(loc=4,frameon=False,handletextpad=0)

# Show and/or print the plot.
fn_plt = "{0}.png".format(fn_acc)
plt.savefig(fn_plt, bbox_inches='tight')
#plt.show()
plt.close()
