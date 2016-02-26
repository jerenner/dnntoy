# -*- coding: utf-8 -*-
"""
Created on Wed Feb  16 2016

dnn2d_googlenet.py

Neural net analysis using GoogleNet

@author: jrenner
"""

import h5py
import numpy as np
import tensorflow as tf

from gnet import GoogleNet
#from gnet_amir import GoogleNet_Amir

fdir = "/home/jmbenlloch/dnntoy/data"
rname = "dnn3d_toy_v1x1x1_r224x224x224"
#rname = "dnn3d_toy_v5x5x5_r200x200x200"

training_run = True;                           # run training step
test_eval_only = False and (not training_run);  # only read test data (cannot be True while training)
    
# Input variables
vox_ext = 112
vox_size = 1
nclass = 2
vox_norm = 1.0      # voxel normalization

num_batches = 3000 #2000
batch_size = 100 #100

ntrain_evts = 750     # number of training evts per dataset
ntest_evts = 250      # number of test events per dataset

aopt_lr = 1.0e-4      # Adam optimizer learning rate
aopt_eps = 1.0e-6     # Adam optimizer epsilon

# Calculated parameters
pdim = int(2 * vox_ext / vox_size)
npix = pdim * pdim
print "Found dim of {0} for {1} pixels.".format(pdim,npix)

# Constructed file names.
fname_si = "{0}/vox_{1}_si.h5".format(fdir,rname)
fname_bg = "{0}/vox_{1}_bg.h5".format(fdir,rname)
fn_saver = "{0}/tfmdl_{1}_pix_{2}_train_{3}.ckpt".format(fdir,rname,npix,ntrain_evts)   # for saving trained network
fn_acc = "{0}/accuracy.dat".format(fdir)

# -----------------------------------------------------------------------------
# Read in all the data.
# -----------------------------------------------------------------------------
dat_si = []; lbl_si = []
dat_bg = []; lbl_bg = []

# Signal
h5f_si = h5py.File(fname_si,'r')
if(test_eval_only):                                  # only read test data 
    ntrk = ntrain_evts; nsi_evts = ntrain_evts + ntest_evts #len(h5f_si)
else:                                                # read all data
    ntrk = 0; nsi_evts = ntrain_evts + ntest_evts #len(h5f_si)
    
print "Reading signal events...";
while(ntrk < nsi_evts):
    trkn = h5f_si['trk{0}'.format(ntrk)]
    darr = np.zeros(3*npix)
    xarr = trkn[0]; yarr = trkn[1]; zarr = trkn[2]; earr = trkn[3]

    # -----------------------
    # Store the projections.
    # -----------------------

    # x-y
    #for xx,yy,ee in zip(xarr,yarr,earr):
    #    darr[int(yy*pdim + xx)] += ee

    # x-y
    for xx,yy,ee in zip(xarr,yarr,earr):
        darr[3*int(yy*pdim + xx)] += ee

    # y-z
    for yy,zz,ee in zip(yarr,zarr,earr):
        darr[3*int(zz*pdim + yy) + 1] += ee
  
    # x-z
    for xx,zz,ee in zip(xarr,zarr,earr):
        darr[3*int(zz*pdim + xx) + 2] += ee
        
    darr *= vox_norm/max(darr)
    dat_si.append(darr)
    lbl_si.append([1, 0])
    ntrk += 1
  
    if(ntrk % int(nsi_evts/100) == 0): print "Read {0}% ...".format(int(100.0*ntrk/nsi_evts));
print "Read {0} signal events".format(len(dat_si))
h5f_si.close()

# Training, background
h5f_bg = h5py.File(fname_bg,'r')
if(test_eval_only):                                  # only read test data 
    ntrk = ntrain_evts; nbg_evts = ntrain_evts + ntest_evts # len(h5f_bg)
else:                                                # read all data
    ntrk = 0; nbg_evts = ntrain_evts + ntest_evts # len(h5f_bg)
    
while(ntrk < nbg_evts):
    trkn = h5f_bg['trk{0}'.format(ntrk)]
    darr = np.zeros(3*npix)
    xarr = trkn[0]; yarr = trkn[1]; zarr = trkn[2]; earr = trkn[3]

    # -----------------------
    # Store the projections.
    # -----------------------

    # x-y
    #for xx,yy,ee in zip(xarr,yarr,earr):
    #    darr[int(yy*pdim + xx)] += ee

    # x-y
    for xx,yy,ee in zip(xarr,yarr,earr):
        darr[3*int(yy*pdim + xx)] += ee
      
    # y-z
    for yy,zz,ee in zip(yarr,zarr,earr):
        darr[3*int(zz*pdim + yy) + 1] += ee

    # x-z
    for xx,zz,ee in zip(xarr,zarr,earr):
        darr[3*int(zz*pdim + xx) + 2] += ee
        
    darr *= vox_norm/max(darr)
    dat_bg.append(darr)
    lbl_bg.append([0, 1])
    ntrk += 1
  
    if(ntrk % int(nbg_evts/100) == 0): print "Read {0}% ...".format(int(100.0*ntrk/nbg_evts));
print "Read {0} background events".format(len(dat_bg))
h5f_bg.close()

# Create one large dataset for test and training datasets.
if(test_eval_only):
    dat_train = []; dat_train_si = []; dat_train_bg = []
    lbl_train = []; lbl_train_si = []; lbl_train_bg = []
    
    dat_test_si = dat_si; dat_test_bg = dat_bg
    lbl_test_si = lbl_si; lbl_test_bg = lbl_bg
else:
    dat_train_si = dat_si[0:ntrain_evts]; dat_train_bg = dat_bg[0:ntrain_evts]
    lbl_train_si = lbl_si[0:ntrain_evts]; lbl_train_bg = lbl_bg[0:ntrain_evts]
    dat_train = np.concatenate((dat_train_si, dat_train_bg))
    lbl_train = np.concatenate((lbl_train_si, lbl_train_bg))
    
    dat_test_si = dat_si[ntrain_evts:]; dat_test_bg = dat_bg[ntrain_evts:]
    lbl_test_si = lbl_si[ntrain_evts:]; lbl_test_bg = lbl_bg[ntrain_evts:]
    
print "Training set has: {0} elements with {1} labels".format(len(dat_train),len(lbl_train))
print "Test set has: {0} elements with {1} labels (si), {2} elements with {3} labels (bg)".format(len(dat_test_si),len(lbl_test_si),len(dat_test_bg),len(lbl_test_bg))

# -----------------------------------------------------------------------------
# Set up the neural network.
# -----------------------------------------------------------------------------

print "Creating placeholders for input and output variables..."
x = tf.placeholder(tf.float32, [batch_size, 3*npix]) # npix])
x_image = tf.reshape(x, [-1,pdim,pdim,3])
y_ = tf.placeholder(tf.float32, [batch_size, nclass])

# Set up the GoogleNet
print "Reading in GoogleNet model..."
net = GoogleNet({'data':x_image})
y_out = net.get_output()
print "Output layer is {0}".format(y_out)

# Set up for training
print "Setting up tf training variables..."
cross_entropy = -tf.reduce_sum(y_*tf.log(y_out + 1.0e-9))
loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
train_step = tf.train.AdamOptimizer(learning_rate=aopt_lr,epsilon=aopt_eps).minimize(cross_entropy)
#train_step = tf.train.GradientDescentOptimizer(0.3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_out,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print "Setting up session..."
sess = tf.Session()
init_op = tf.initialize_all_variables()
sess.run(init_op)

# Load in the previously trained data.
net.load('gnet.npy', sess)

# Create a saver to save the DNN.
saver = tf.train.Saver()

# Restore saved variables if the option is set.
if(not training_run):
    #print "Restoring variables read from file {0} ...".format(fn_saver)
    #saver.restore(sess,fn_saver)
    net.load('gnet_amir.npy',sess)
else:

    # Train the NN in batches.
    nbatch_dat = len(dat_train) / batch_size  # number of batches that fit in the data length
    lacc_tr = []; lacc_si = []; lacc_bg = []
    for bnum in range(num_batches):
        
        print "Training batch {0} of {1}".format(bnum,num_batches)
        
        # Shuffle the data if we reached the end of the loop.
        #if(bnum == 0 or (bnum*batch_size < len(dat_train) and (bnum+1)*batch_size >= len(dat_train))):
        if(bnum % nbatch_dat == 0):
        
          print "Shuffling data..."
          perm = np.arange(len(dat_train))
          np.random.shuffle(perm)
          dat_train = dat_train[perm]
          lbl_train = lbl_train[perm]
    
        # Be sure not to go over the number of batches that fit in the data length.
        btemp = bnum % nbatch_dat
        
        batch_xs = dat_train[btemp*batch_size:(btemp + 1)*batch_size,:]
        batch_ys = lbl_train[btemp*batch_size:(btemp + 1)*batch_size,:]
        _, loss_val = sess.run([train_step, loss], feed_dict={x: batch_xs, y_: batch_ys})
        print "Got loss value of {0}".format(loss_val)

        # Check accuracy of this run.
        acc_tr = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
        acc_si = sess.run(accuracy, feed_dict={x: dat_test_si[0:batch_size], y_: lbl_test_si[0:batch_size]})
        acc_bg = sess.run(accuracy, feed_dict={x: dat_test_bg[0:batch_size], y_: lbl_test_bg[0:batch_size]})
        lacc_tr.append(acc_tr); lacc_si.append(acc_si); lacc_bg.append(acc_bg)
        print "-- Training accuracy = {0}; Validation accuracy = {1} (si), {2} (bg)".format(acc_tr,acc_si,acc_bg)
    
    # Save the trained model.
    print "Saving trained model to: {0}".format(fn_saver)
    save_path = saver.save(sess, fn_saver)
    
    # Save the accuracy lists.
    f_acc = open(fn_acc,"w")
    f_acc.write("# train v_si v_bg -- one entry per iteration\n")
    for atr,asi,abg in zip(lacc_tr,lacc_si,lacc_bg):
        f_acc.write("{0} {1} {2}\n".format(atr,asi,abg))
    f_acc.close()

# Evaluate the performance.
if(not training_run):
    print "On test signal data:"
    print sess.run(accuracy, feed_dict={x: dat_test_si[0:200], y_: lbl_test_si[0:200]})
    print "On test background data:"
    print sess.run(accuracy, feed_dict={x: dat_test_bg[0:200], y_: lbl_test_bg[0:200]})

    if(not test_eval_only):
        print "On training signal data:"
        print sess.run(accuracy, feed_dict={x: dat_train_si[0:200], y_: lbl_train_si[0:200]})
        print "On training background data:"
        print sess.run(accuracy, feed_dict={x: dat_train_bg[0:200], y_: lbl_train_bg[0:200]})
