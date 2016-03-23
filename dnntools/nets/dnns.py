"""

Configuration for different nets.

"""

import h5py
import numpy as np
import tensorflow as tf


# -----------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Helper methods.
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
def getNet(name,x_image):

    if(name == "MNISTadv"):
        return MNISTadv_net(x_image)

# -----------------------------------------------------------------------------
# DNN definitions
# -----------------------------------------------------------------------------
def MNISTadv_net(x_image):

    # Get the number of pixels in one dimension.
    pdim = x_image.get_shape()[1].value

    # First convolutional layer
    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])
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
    keep_prob = 0.4 #tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Softmax readout
    W_fc2 = weight_variable([1024, 2])
    b_fc2 = bias_variable([2])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    return y_conv
