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
import os

from gnet import GoogleNet
#from gnet_amir import GoogleNet_Amir

fdir = "/home/jmbenlloch/dnntoy/data"
rname = "dnn3d_NEXT100_Paolina222_v2x2x2_r200x200x200"
#rname = "dnn3d_toy_v5x5x5_r200x200x200"

training_run = True;                           # run training step
train_init = False;                             # if true, train from net with standard pre-training; if false, read in a previously trained net
    
# Input variables
vox_ext = 100
vox_size = 2
nclass = 2
vox_norm = 1.0      # voxel normalization

ntrain_evts = 15000     # number of training evts per dataset
#ntest_evts = 2000       # number of test events per dataset
num_epochs = 100         # total number of epochs to train 
epoch_blk_size = 1      # number of epochs to run per block (before reading new dataset); set equal to num_epochs unless data to be read in multiple blocks
dtblk_size = 5000    # number of signal and background events per training block
batch_size = 100       # training batch size
acc_blk = 2000          # number of events to use in checking for accuracy
print "Params:\n ntrain_evts = {0}, num_epochs = {1}, epoch_blk_size = {2}, dtblk_size = {3}, batch_size = {4}, acc_blk = {5}".format(ntrain_evts,num_epochs,epoch_blk_size,dtblk_size,batch_size,acc_blk);

aopt_lr = 1.0e-4      # Adam optimizer learning rate
aopt_eps = 1.0e-6     # Adam optimizer epsilon

# Checks on parameters
if(ntrain_evts % dtblk_size != 0):
    print "ERROR: ntrain_evts must be evenly divisible by dtblk_size."; exit()
if(num_epochs % epoch_blk_size != 0):
    print "ERROR: num_epochs must be evenly divisible by epoch_blk_size..."; exit()
if(ntrain_evts % batch_size != 0):
    print "ERROR: ntrain_evts must be evenly divisible by batch_size..."; exit()
if(acc_blk % batch_size != 0):
    print "ERROR: acc_blk must be evenly divisible by batch_size..."; exit()

# Calculated parameters
batches_per_epoch = int(2*dtblk_size/batch_size)
num_epoch_blks = num_epochs / epoch_blk_size
num_dt_blks = ntrain_evts / dtblk_size
pdim = int(2 * vox_ext / vox_size)
npix = pdim * pdim
print "Found dim of {0} for {1} pixels.".format(pdim,npix)

# Constructed file names.
fname_si = "{0}/vox_{1}_si.h5".format(fdir,rname)
fname_bg = "{0}/vox_{1}_bg.h5".format(fdir,rname)
fn_saver = "{0}/models/{1}/tfmdl_{2}_pix_{3}.ckpt".format(fdir,rname,rname,npix)   # for saving trained network
fn_acc = "{0}/acc/{1}/accuracy_{2}.dat".format(fdir,rname,rname)
fn_prob = "{0}/acc/{1}/prob_{2}".format(fdir,rname,rname)

if(not os.path.isdir("{0}/acc/{1}".format(fdir,rname))): os.mkdir("{0}/acc/{1}".format(fdir,rname))
if(not os.path.isdir("{0}/models/{1}".format(fdir,rname))): os.mkdir("{0}/models/{1}".format(fdir,rname))
# ---------------------------------------------------------------------------------------
# Function definitions
# ---------------------------------------------------------------------------------------

# Evaluate the performance.
def eval_performance(fsummary,epoch,sess,loss,y_out,dat_train_si,dat_train_bg,dat_test_si,dat_test_bg):

    # ----------------------------
    # Evaluate the training data.
    # ----------------------------
    f_prob_train = open("{0}_train_ep{1}.dat".format(fn_prob,epoch),"w")
    acc_tr_si = 0.; acc_tr_bg = 0.; lval_tr_si = 0.; lval_tr_bg = 0.
    # Signal
    print "-- TRAINING data: signal"
    nevt = 0; nbatches = 0
    while(nevt < acc_blk):
        ltemp,ytemp = sess.run([loss,y_out],feed_dict={x: dat_train_si[nevt:nevt+batch_size], y_: lbl_train_si[nevt:nevt+batch_size]})
        for y0 in ytemp[:,0]:
            if(y0 > 0.5): acc_tr_si += 1
        lval_tr_si += ltemp
        for y0,y1 in zip(ytemp[:,0],ytemp[:,1]): f_prob_train.write("{0} {1} {2}\n".format(1,y0,y1))
        nevt += batch_size; nbatches += 1

    # Background
    print "-- TRAINING data: background"
    nevt = 0; nbatches = 0
    while(nevt < acc_blk):
        ltemp,ytemp = sess.run([loss,y_out],feed_dict={x: dat_train_bg[nevt:nevt+batch_size], y_: lbl_train_bg[nevt:nevt+batch_size]})
        for y1 in ytemp[:,1]:
            if(y1 > 0.5): acc_tr_bg += 1
        lval_tr_bg += ltemp
        for y0,y1 in zip(ytemp[:,0],ytemp[:,1]): f_prob_train.write("{0} {1} {2}\n".format(0,y0,y1))
        nevt += batch_size; nbatches += 1
    f_prob_train.close()

    acc_tr_si /= acc_blk; acc_tr_bg /= acc_blk
    lval_tr_si /= nbatches; lval_tr_bg /= nbatches

    # ---------------------------
    # Evaluate the test data.
    # ---------------------------
    f_prob_test = open("{0}_test_ep{1}.dat".format(fn_prob,epoch),"w")
    acc_te_si = 0.; acc_te_bg = 0.; lval_te_si = 0.; lval_te_bg = 0.
    # Signal
    print "-- TEST data: signal"
    nevt = 0; nbatches = 0
    while(nevt < acc_blk):
        ltemp,ytemp = sess.run([loss,y_out],feed_dict={x: dat_test_si[nevt:nevt+batch_size], y_: lbl_test_si[nevt:nevt+batch_size]})
        for y0 in ytemp[:,0]:
            if(y0 > 0.5): acc_te_si += 1
        lval_te_si += ltemp
        for y0,y1 in zip(ytemp[:,0],ytemp[:,1]): f_prob_test.write("{0} {1} {2}\n".format(1,y0,y1))
        nevt += batch_size; nbatches += 1

    # Background
    print "-- TEST data: background"
    nevt = 0; nbatches = 0
    while(nevt < acc_blk):
        ltemp,ytemp = sess.run([loss,y_out],feed_dict={x: dat_test_bg[nevt:nevt+batch_size], y_: lbl_test_bg[nevt:nevt+batch_size]})
        for y1 in ytemp[:,1]:
            if(y1 > 0.5): acc_te_bg += 1
        lval_te_bg += ltemp
        for y0,y1 in zip(ytemp[:,0],ytemp[:,1]): f_prob_test.write("{0} {1} {2}\n".format(0,y0,y1))
        nevt += batch_size; nbatches += 1
    f_prob_test.close()

    acc_te_si /= acc_blk; acc_te_bg /= acc_blk
    lval_te_si /= nbatches; lval_te_bg /= nbatches

    # Write to the final summary file.
    fsummary.write("{0} {1} {2} {3} {4} {5} {6} {7} {8}\n".format(epoch,acc_tr_si,acc_tr_bg,lval_tr_si,lval_tr_bg,acc_te_si,acc_te_bg,lval_te_si,lval_te_bg))

# Read in all the data from evt_start to evt_end-1.
# - assumes that events are labeled as trkX in the hdf5 file, where X is the event number
def read_data(h5f_si,h5f_bg,evt_start,evt_end):

    nevts = evt_end - evt_start

    print "-- read_data: Reading events from {0} to {1} for signal and background".format(evt_start,evt_end)

    # Set up the data arrays.
    dat_si = np.zeros([nevts,3*npix]); lbl_si = np.zeros([nevts,2])
    dat_bg = np.zeros([nevts,3*npix]); lbl_bg = np.zeros([nevts,2])

    # Read in all events from the hdf5 files.
    ntrk = 0
    while(ntrk < nevts):

        # Read the signal event.
        trkn_si = h5f_si['trk{0}'.format(evt_start+ntrk)]
        xarr = trkn_si[0]; yarr = trkn_si[1]; zarr = trkn_si[2]; earr = trkn_si[3]
        for xx,yy,ee in zip(xarr,yarr,earr): dat_si[ntrk][3*int(yy*pdim + xx)] += ee         # x-y projection
        for yy,zz,ee in zip(yarr,zarr,earr): dat_si[ntrk][3*int(zz*pdim + yy) + 1] += ee     # y-z projection
        for xx,zz,ee in zip(xarr,zarr,earr): dat_si[ntrk][3*int(zz*pdim + xx) + 2] += ee     # x-z projection
        dat_si[ntrk] *= vox_norm/max(dat_si[ntrk])
        lbl_si[ntrk][0] = 1    # set the label to signal

        # Read the background event.
        trkn_bg = h5f_bg['trk{0}'.format(evt_start+ntrk)]
        xarr = trkn_bg[0]; yarr = trkn_bg[1]; zarr = trkn_bg[2]; earr = trkn_bg[3]
        for xx,yy,ee in zip(xarr,yarr,earr): dat_bg[ntrk][3*int(yy*pdim + xx)] += ee         # x-y projection
        for yy,zz,ee in zip(yarr,zarr,earr): dat_bg[ntrk][3*int(zz*pdim + yy) + 1] += ee     # y-z projection
        for xx,zz,ee in zip(xarr,zarr,earr): dat_bg[ntrk][3*int(zz*pdim + xx) + 2] += ee     # x-z projection
        dat_bg[ntrk] *= vox_norm/max(dat_bg[ntrk])
        lbl_bg[ntrk][1] = 1    # set the label to signal

        ntrk += 1

    # Return the data and labels.
    return (dat_si,lbl_si,dat_bg,lbl_bg)

# Set up the neural network.
def net_setup():

    print "\n\n-- net_setup():  SETTING UP NETWORK --"

    print "Creating placeholders for input and output variables..."
    x = tf.placeholder(tf.float32, [batch_size, 3*npix]) # npix])
    x_image = tf.reshape(x, [-1,pdim,pdim,3])
    y_ = tf.placeholder(tf.float32, [batch_size, nclass])

    # Set up the GoogleNet
    print "Reading in GoogLeNet model..."
    net = GoogleNet({'data':x_image})
    y_out = net.get_output()
    print "Output layer is {0}".format(y_out)

    # Set up for training
    print "Setting up tf training variables..."
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_out + 1.0e-9))
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    train_step = tf.train.AdamOptimizer(learning_rate=aopt_lr,epsilon=aopt_eps).minimize(cross_entropy)
    #train_step = tf.train.GradientDescentOptimizer(0.3).minimize(cross_entropy)
    #correct_prediction = tf.equal(tf.argmax(y_out,1), tf.argmax(y_,1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print "Setting up session..."
    sess = tf.Session()
    init_op = tf.initialize_all_variables()
    sess.run(init_op)

    # Create a saver to save the DNN.
    saver = tf.train.Saver()

    # Load in the previously trained data.
    if(train_init):
        print "Loading in previously trained GoogLeNet parameters..."
        net.load('gnet.npy', sess)
    else:
        print "Restoring previously trained net from file {0}".format(fn_saver)
        saver.restore(sess,fn_saver) 

    return (sess,train_step,loss,x,y_,y_out,saver)

# -----------------------------------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------------------------------

# Set up the DNN.
(sess,train_step,loss,x,y_,y_out,saver) = net_setup()

# Open the relevant files.
f_acc = open(fn_acc,'w')
h5f_si = h5py.File(fname_si,'r')
h5f_bg = h5py.File(fname_bg,'r')

# Read in a validation set for short checks on accuracy.
dat_val_si = np.zeros([batch_size,3*npix]); lbl_val_si = np.zeros([batch_size,2])
dat_val_bg = np.zeros([batch_size,3*npix]); lbl_val_bg = np.zeros([batch_size,2])
(dat_val_si[:],lbl_val_si[:],dat_val_bg[:],lbl_val_bg[:]) = read_data(h5f_si,h5f_bg,ntrain_evts,ntrain_evts+batch_size)

# Iterate over all epoch blocks.
for eblk in range(num_epoch_blks):

    print "\n\n**EPOCH BLOCK {0}".format(eblk)

    # Iterate over data blocks.
    for dtblk in range(num_dt_blks):

        print "- DATA BLOCK {0}".format(dtblk)

        # Read in the data.
        evt_start = dtblk*dtblk_size
        evt_end = (dtblk+1)*dtblk_size
        dat_train = np.zeros([2*dtblk_size,3*npix])
        lbl_train = np.zeros([2*dtblk_size,2])
        (dat_train[0:dtblk_size],lbl_train[0:dtblk_size],dat_train[dtblk_size:],lbl_train[dtblk_size:]) = read_data(h5f_si,h5f_bg,evt_start,evt_end)

        # Iterate over epochs within the block.
        for ep in range(epoch_blk_size):

            print "-- EPOCH {0} of block size {1}".format(ep,epoch_blk_size)

            # Shuffle the data.
            print "--- Shuffling data..."
            perm = np.arange(len(dat_train))
            np.random.shuffle(perm)
            dat_train = dat_train[perm]
            lbl_train = lbl_train[perm]

            # Train the NN in batches.
            for bnum in range(batches_per_epoch):

                print "--- Training batch {0} of {1}".format(bnum,batches_per_epoch)

                batch_xs = dat_train[bnum*batch_size:(bnum + 1)*batch_size,:]
                batch_ys = lbl_train[bnum*batch_size:(bnum + 1)*batch_size,:]
                _, loss_val = sess.run([train_step, loss], feed_dict={x: batch_xs, y_: batch_ys})
                print "--- Got loss value of {0}".format(loss_val)

            # Run a short accuracy check.
            acc_train = 0.; acc_test_si = 0.; acc_test_bg = 0.
            ltemp,ytemp = sess.run([loss,y_out],feed_dict={x: dat_train[0:batch_size], y_: lbl_train[0:batch_size]})
            for yin,yout in zip(lbl_train[0:batch_size],ytemp):
                if(np.argmax(yin) == np.argmax(yout)): acc_train += 1
            acc_train /= batch_size
            ltemp,ytemp = sess.run([loss,y_out],feed_dict={x: dat_val_si, y_: lbl_val_si})
            for yin,yout in zip(lbl_val_si,ytemp):
                if(np.argmax(yin) == np.argmax(yout)): acc_test_si += 1
            acc_test_si /= batch_size
            ltemp,ytemp = sess.run([loss,y_out],feed_dict={x: dat_val_bg, y_: lbl_val_bg})
            for yin,yout in zip(lbl_val_bg,ytemp):
                if(np.argmax(yin) == np.argmax(yout)): acc_test_bg += 1
            acc_test_bg /= batch_size
            print "--- Training accuracy = {0}; Test signal accuracy = {1}, Test background accuracy = {2}".format(acc_train,acc_test_si,acc_test_bg)

    # Calculate the number of epochs run.
    epoch = (eblk+1)*epoch_blk_size
    print "Checking accuracy after {0} epochs".format(epoch)

    # Read in the data to be used in the accuracy check.
    dat_train_si = np.zeros([acc_blk,3*npix]); lbl_train_si = np.zeros([acc_blk,2])
    dat_train_bg = np.zeros([acc_blk,3*npix]); lbl_train_bg = np.zeros([acc_blk,2])
    (dat_train_si[:],lbl_train_si[:],dat_train_bg[:],lbl_train_bg[:]) = read_data(h5f_si,h5f_bg,0,acc_blk)

    dat_test_si = np.zeros([acc_blk,3*npix]); lbl_test_si = np.zeros([acc_blk,2])
    dat_test_bg = np.zeros([acc_blk,3*npix]); lbl_test_bg = np.zeros([acc_blk,2])
    (dat_test_si[:],lbl_test_si[:],dat_test_bg[:],lbl_test_bg[:]) = read_data(h5f_si,h5f_bg,ntrain_evts,ntrain_evts+acc_blk)

    # Run the accuracy check.
    eval_performance(f_acc,epoch,sess,loss,y_out,dat_train_si,dat_train_bg,dat_test_si,dat_test_bg)

    # Save the trained model.
    print "Saving trained model to: {0}".format(fn_saver)
    save_path = saver.save(sess, fn_saver)

# Close the relevant files.
f_acc.close()
h5f_si.close()
h5f_bg.close()
