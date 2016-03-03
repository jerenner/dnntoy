from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy import ndimage
from six.moves import cPickle as pickle
import imagehash
from PIL import Image
import os
import sys
from os.path import isfile, join

from Util import *

from IPython.display import display
from sklearn.linear_model import LogisticRegression

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.
train_images = 45000
test_images = 1800


def wait():
  """
  Convenience rename of raw_input
  """
  raw_input("Press a key...")

def plot_image(dataset,indx):
  """
  Plot image indx in dataset
  """
  image = dataset[indx]
  print("shape of image =",image.shape)
  img = (image + pixel_depth / 2) / pixel_depth
  plt.imshow(img)
  plt.show()

def print_hash(hset,indx):
  """
  print hash indx in hashset
  """
  print("for image ={}: hash ={} ".format(
      indx,hset[indx]))

def print_label(lbl,indx):
  """
  print lbl indx in labels
  """
  print("for image ={}: label ={} ".format(
      indx,lbl[indx]))

def compare_hash(hset1,hset2,indx):
  """
  Compare two hashes 
  """
  if hset1[indx] == hset2[indx]:
    print("hashes matches for image ={}: hash ={} ".format(
      indx,hset1[indx]))
  else:
    print("hash mitmach for image ={}".format(indx))
    print("hash1 ={}, hash2 ={}".format(hset1[indx],hset2[indx]))

###  

def display(pickle_file, sample):
  """
  displays images in sample and check hashes for pickle_file
  """
  dataset, hashset =load_data(pickle_file)
  
  for indx in sample:
    plot_image(dataset,indx)
    print_hash(hashset,indx)

def display_merged(pickle_file, dset, sample):
  """
  displays images in sample and check hashes for pickle_file
  dset = train, valid or test
  """
  datadict = load_merged_data(pickle_file)  
  print("loaded datadict: keys =",datadict.keys()) 

  train_dataset= datadict['train_dataset']
  train_hashset= datadict['train_hashset'] 
  train_labels = datadict['train_labels'] 
  valid_dataset= datadict['valid_dataset']
  valid_hashset= datadict['valid_hashset']
  valid_labels = datadict['valid_labels']
  test_dataset= datadict['test_dataset']
  test_hashset= datadict['test_hashset']
  test_labels = datadict['test_labels']
  
  for indx in sample:
    if dset == 'train':
      plot_image(train_dataset,indx)
      print_hash(train_hashset,indx)
      print_label(train_labels,indx)
    elif dset == 'valid':
      plot_image(valid_dataset,indx)
      print_hash(valid_hashset,indx)
      print_label(valid_labels,indx)
    elif dset == 'test':
      plot_image(test_dataset,indx)
      print_hash(test_hashset,indx)
      print_label(test_labels,indx)
    else:
      print("invalid data set: acceptable data sets are train, valid, test")
      sys.exit()


# def display(pickle_file,number_of_images=5):
#   """
#   displays first images and check hashes for pickle_file
#   """
#   print("reading pickle file = {}".format(pickle_file))
#   print("reading {} images".format(number_of_images))

#   dataset, hashset =load_data(pickle_file)

#   for indx in range(number_of_images):
#     plot_image(dataset,indx)
#     print_hash(hashset,indx)

def load_data(pickle_file):
  """
  pickle back the data as a dictionary 
  datadict = {
      'letter_dataset': dataset,
      'letter_hashset': hashset
      }
  """
  try:
    f = open(pickle_file, 'rb')
    datadict = pickle.load(f)
    f.close()
    dataset= datadict['letter_dataset']
    hashset= datadict['letter_hashset']
    return dataset, hashset
  except Exception as e:
    print('Unable to load data from', pickle_file, ':', e)
    raise

def load_merged_data(pickle_file):
  """
  pickle back the data as a dictionary 
  datadict = {
      'train_dataset': train_dataset,
      'train_hashset': train_hashset,
      'train_labels': train_labels,
      'valid_dataset': valid_dataset,
      'valid_hashset': valid_hashset,
      'valid_labels': valid_labels,
      'test_dataset': test_dataset,
      'test_hashset': test_hashset,
      'test_labels': test_labels
      }
  """
  try:
    f = open(pickle_file, 'rb')
    datadict = pickle.load(f)
    f.close()
    return datadict
  except Exception as e:
    print('Unable to load data from', pickle_file, ':', e)
    raise
