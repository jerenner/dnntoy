# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 18:24:11 2016

@author: jrenner
"""

import h5py
import numpy as np
import tensorflow as tf

fdir = "/Users/jrenner/IFIC/dnn/tracks/data"
rname = "dnn2d_4mm"

fname_train_si = "{0}/vox_{1}_train_si.h5".format(fdir,rname)
fname_train_bg = "{0}/vox_{1}_train_bg.h5".format(fdir,rname)

fname_test_si = "{0}/vox_{1}_test_si.h5".format(fdir,rname)
fname_test_bg = "{0}/vox_{1}_test_bg.h5".format(fdir,rname)
    
# Input variables
vox_ext = 112
vox_size = 1
nclass = 2

num_batches = 4000
batch_size = 100

# Calculated parameters
pdim = int(2 * vox_ext / vox_size)
npix = pdim * pdim
print "Found dim of {0} for {1} pixels.".format(pdim,npix)

# -----------------------------------------------------------------------------
# Read in all the data.
# -----------------------------------------------------------------------------
dat_train_si = []; lbl_train_si = []
dat_train_bg = []; lbl_train_bg = []

dat_test_si = []; lbl_test_si = []
dat_test_bg = []; lbl_test_bg = []

# Training, signal
h5f_train_si = h5py.File(fname_train_si,'r')
ntrk = 0
while(ntrk < len(h5f_train_si)):
  trkn = h5f_train_si['trk{0}'.format(ntrk)]
  darr = np.zeros(npix)
  xarr = trkn[0]; yarr = trkn[1]; earr = trkn[2]
  for xx,yy,ee in zip(xarr,yarr,earr):
      darr[int(yy*pdim + xx)] = ee
  darr *= 1./max(darr)
  dat_train_si.append(darr)
  lbl_train_si.append([1, 0])
  ntrk += 1
print "Read {0} signal training events".format(len(dat_train_si))
h5f_train_si.close()

# Training, background
h5f_train_bg = h5py.File(fname_train_bg,'r')
ntrk = 0
while(ntrk < len(h5f_train_bg)):
  trkn = h5f_train_bg['trk{0}'.format(ntrk)]
  darr = np.zeros(npix)
  xarr = trkn[0]; yarr = trkn[1]; earr = trkn[2]
  for xx,yy,ee in zip(xarr,yarr,earr):
      darr[int(yy*pdim + xx)] = ee
  darr *= 1./max(darr)
  dat_train_bg.append(darr)
  lbl_train_bg.append([0, 1])
  ntrk += 1
print "Read {0} background training events".format(len(dat_train_bg))
h5f_train_bg.close()

# Test, signal
h5f_test_si = h5py.File(fname_test_si,'r')
ntrk = 0
while(ntrk < len(h5f_test_si)):
  trkn = h5f_test_si['trk{0}'.format(ntrk)]
  darr = np.zeros(npix)
  xarr = trkn[0]; yarr = trkn[1]; earr = trkn[2]
  for xx,yy,ee in zip(xarr,yarr,earr):
      darr[int(yy*pdim + xx)] = ee
  darr *= 1./max(darr)
  dat_test_si.append(darr)
  lbl_test_si.append([1, 0])
  ntrk += 1
print "Read {0} signal test events".format(len(dat_test_si))
h5f_test_si.close()

# Test, background
h5f_test_bg = h5py.File(fname_test_bg,'r')
ntrk = 0
while(ntrk < len(h5f_test_bg)):
  trkn = h5f_test_bg['trk{0}'.format(ntrk)]
  darr = np.zeros(npix)
  xarr = trkn[0]; yarr = trkn[1]; earr = trkn[2]
  for xx,yy,ee in zip(xarr,yarr,earr):
      darr[int(yy*pdim + xx)] = ee
  darr *= 1./max(darr)
  dat_test_bg.append(darr)
  lbl_test_bg.append([0, 1])
  ntrk += 1
print "Read {0} background test events".format(len(dat_test_bg))
h5f_test_bg.close()

# -----------------------------------------------------------------------------
# Define helper methods.
# -----------------------------------------------------------------------------

# Create one large training dataset.
dat_train = np.concatenate((dat_train_si, dat_train_bg))
lbl_train = np.concatenate((lbl_train_si, lbl_train_bg))

dat_test = np.concatenate((dat_test_si, dat_test_bg))
lbl_test = np.concatenate((lbl_test_si, lbl_test_bg))

# Set up the neural network.
x = tf.placeholder(tf.float32, [None, npix])
W = tf.Variable(tf.zeros([npix, nclass]))
b = tf.Variable(tf.zeros([nclass]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, nclass])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# Train the NN in batches.
nbatch_dat = len(dat_train) / batch_size  # number of batches that fit in the data length
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
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Evaluate the performance.
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print "On test data:"
print sess.run(accuracy, feed_dict={x: dat_test, y_: lbl_test})
print "On training data:"
print sess.run(accuracy, feed_dict={x: dat_train, y_: lbl_train})