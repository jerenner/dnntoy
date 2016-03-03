# -*- coding: utf-8 -*-
"""
Created on Wed Feb  16 2016

dnn2d_conv3p.py

2D convolutional neural network using 3 projections

Uses code from TensorFlow tutorial:
https://www.tensorflow.org/versions/0.6.0/tutorials/mnist/pros/index.html

@author: jrenner
"""

import h5py
import numpy as np
import tensorflow as tf

fdir = "/home/jrenner/dnn/data"
rname = "dnn3d_NEXT100_Paolina222_v2x2x2_r200x200x200"

training_run = False;                           # run training step
test_eval_only = False and (not training_run);  # only read test data (cannot be True while training)
    
# Input variables
vox_ext = 100
vox_size = 2
nclass = 2

num_batches = 1000 #2000
batch_size = 100 #100
print "Number of batches = {0}; batch size = {1}".format(num_batches,batch_size)

ntrain_evts = 1500     # number of training evts per dataset
ntest_evts = 500       # number of test events per dataset
print "Number of training events = {0}; number of test events = {1}".format(ntrain_evts,ntest_evts)

# Calculated parameters
pdim = int(2 * vox_ext / vox_size)
npix = pdim * pdim
print "Found dim of {0} for {1} pixels.".format(pdim,npix)

# Constructed file names.
fname_si = "{0}/vox_{1}_si.h5".format(fdir,rname)
fname_bg = "{0}/vox_{1}_bg.h5".format(fdir,rname)
fn_saver = "{0}/models/tfmdl_{1}_pix_{2}_train_{3}.ckpt".format(fdir,rname,npix,ntrain_evts)   # for saving trained network
fn_acc = "{0}/acc/accuracy_{1}_pix_{2}_train_{3}.dat".format(fdir,rname,npix,ntrain_evts)

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
    for xx,yy,ee in zip(xarr,yarr,earr):
        darr[3*int(yy*pdim + xx)] += ee
      
    # x-z
    for xx,zz,ee in zip(xarr,zarr,earr):
        darr[3*int(zz*pdim + xx) + 1] += ee
        
    # y-z
    for yy,zz,ee in zip(yarr,zarr,earr):
        darr[3*int(zz*pdim + yy) + 2] += ee
        
    darr *= 1./max(darr)
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
    for xx,yy,ee in zip(xarr,yarr,earr):
        darr[3*int(yy*pdim + xx)] += ee
      
    # x-z
    for xx,zz,ee in zip(xarr,zarr,earr):
        darr[3*int(zz*pdim + xx) + 1] += ee
        
    # y-z
    for yy,zz,ee in zip(yarr,zarr,earr):
        darr[3*int(zz*pdim + yy) + 2] += ee
        
    darr *= 1./max(darr)
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
# Define helper methods.
# -----------------------------------------------------------------------------
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
  
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# -----------------------------------------------------------------------------
# Set up the neural network.
# -----------------------------------------------------------------------------

x = tf.placeholder(tf.float32, [None, 3*npix])
y_ = tf.placeholder(tf.float32, [None, nclass])

# First convolutional layer
W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,pdim,pdim,3])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Densely connected layer
new_dim = int(pdim / 4)
W_fc1 = weight_variable([new_dim * new_dim * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, new_dim*new_dim*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Softmax readout
W_fc2 = weight_variable([1024, nclass])
b_fc2 = bias_variable([nclass])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Set up for training
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv + 1.0e-9))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess = tf.Session()
init_op = tf.initialize_all_variables()
sess.run(init_op)

# Create a saver to save the DNN.
saver = tf.train.Saver()

# Restore saved variables if the option is set.
if(not training_run):
    print "Restoring variables read from file {0} ...".format(fn_saver)
    saver.restore(sess,fn_saver)
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
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

        # Check accuracy of this run.
        acc_tr = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
        acc_si = sess.run(accuracy, feed_dict={x: dat_test_si[0:100], y_: lbl_test_si[0:100], keep_prob: 1.0})
        acc_bg = sess.run(accuracy, feed_dict={x: dat_test_bg[0:100], y_: lbl_test_bg[0:100], keep_prob: 1.0})
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

    # Evaluate the training data.
    if(not test_eval_only):

        nevt = 0
        acc_si = 0; acc_bg = 0
        while(nevt < ntrain_evts):
            nevts_i = batch_size
            if(nevt > (ntrain_evts - batch_size)):
                nevts_i = ntrain_evts - nevt
            nevt_end = nevt + nevts_i
            print "-- Evaluating training data for events {0} to {1}:".format(nevt,nevt_end)
            acc_si += sess.run(accuracy, feed_dict={x: dat_train_si[nevt:nevt_end], y_: lbl_train_si[nevt:nevt_end], keep_prob: 1.0})*nevts_i
            acc_bg += sess.run(accuracy, feed_dict={x: dat_train_bg[nevt:nevt_end], y_: lbl_train_bg[nevt:nevt_end], keep_prob: 1.0})*nevts_i
            nevt += batch_size
        print "On training signal data, accuracy = {0}".format(1.0*acc_si/ntrain_evts)
        print "On training background data, accuracy = {0}".format(1.0*acc_bg/ntrain_evts)

    # Evaluate the test data.
    nevt = 0
    acc_si = 0; acc_bg = 0
    while(nevt < ntest_evts):
        nevts_i = batch_size
        if(nevt > (ntest_evts - batch_size)):
            nevts_i = ntest_evts - nevt
        nevt_end = nevt + nevts_i
        print "-- Evaluating test data for events {0} to {1}:".format(nevt,nevt_end)
        acc_si += sess.run(accuracy, feed_dict={x: dat_test_si[nevt:nevt_end], y_: lbl_test_si[nevt:nevt_end], keep_prob: 1.0})*nevts_i
        acc_bg += sess.run(accuracy, feed_dict={x: dat_test_bg[nevt:nevt_end], y_: lbl_test_bg[nevt:nevt_end], keep_prob: 1.0})*nevts_i
        nevt += batch_size
    print "On test signal data, accuracy = {0}".format(1.0*acc_si/ntest_evts)
    print "On test background data, accuracy = {0}".format(1.0*acc_bg/ntest_evts)
