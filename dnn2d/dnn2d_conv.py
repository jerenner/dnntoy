# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 2016

Uses code from TensorFlow tutorial:
https://www.tensorflow.org/versions/0.6.0/tutorials/mnist/pros/index.html

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
vox_ext = 96
vox_size = 4
nclass = 2

num_batches = 2000
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


# Create one large training dataset.
dat_train = np.concatenate((dat_train_si, dat_train_bg))
lbl_train = np.concatenate((lbl_train_si, lbl_train_bg))

# Create one large test dataset.
dat_test = np.concatenate((dat_test_si, dat_test_bg))
lbl_test = np.concatenate((lbl_test_si, lbl_test_bg))

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

x = tf.placeholder(tf.float32, [None, npix])
y_ = tf.placeholder(tf.float32, [None, nclass])
#W = tf.Variable(tf.zeros([npix, nclass]))
#b = tf.Variable(tf.zeros([nclass]))

# First convolutional layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,pdim,pdim,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Densely connected layer
W_fc1 = weight_variable([12 * 12 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 12*12*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Softmax readout
W_fc2 = weight_variable([1024, nclass])
b_fc2 = bias_variable([nclass])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Set up for training
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess = tf.Session()
sess.run(tf.initialize_all_variables())

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
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

# Evaluate the performance.
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print "On test data:"
print sess.run(accuracy, feed_dict={x: dat_test, y_: lbl_test, keep_prob: 1.0})
print "On training data:"
print sess.run(accuracy, feed_dict={x: dat_train, y_: lbl_train, keep_prob: 1.0})