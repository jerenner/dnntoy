from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os

from os.path import isfile, join

from Util import *

import matplotlib.image as mpimg
from scipy import ndimage
from six.moves import cPickle as pickle


image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

train_folders = ['notMNIST_large/A', 'notMNIST_large/B', 
'notMNIST_large/C', 'notMNIST_large/D', 'notMNIST_large/E', 
'notMNIST_large/F', 'notMNIST_large/G', 'notMNIST_large/H', 
'notMNIST_large/I', 'notMNIST_large/J']
test_folders =['notMNIST_small/A', 'notMNIST_small/B', 
'notMNIST_small/C', 'notMNIST_small/D', 'notMNIST_small/E', 
'notMNIST_small/F', 'notMNIST_small/G', 'notMNIST_small/H', 
'notMNIST_small/I', 'notMNIST_small/J']

train_folders_test = ['notMNIST_large/A']
test_folders_test =['notMNIST_small/A']

SampleTrain =np.random.randint(0, 45000, size=5)
SampleTest =np.random.randint(0, 1800, size=5)

def load_image(path_to_image_file):
  """

  loads the dataset from each file

  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32) #3D array of floats
  
  """

  dataset = pickle.load( open( path_to_image_file, "rb" ) )
  return dataset

def check_size(dataset):
  return dataset.shape[0]
  

def plot_image(dataset,indx):
  image = dataset[indx]
  print("shape of image =",image.shape)
  img = (image + pixel_depth / 2) / pixel_depth
  plt.imshow(img)
  plt.show()


def sample_plot(folders,sample):
  print("Sample  images =", sample)
  for name in folders:
    path = name+'.pickle'
    print("loading %s dataset"%path)
    dataset =load_image(path)
    
    n=0
    for indx in sample:
      print("plotting image = %d index = %d"%(n,indx))
      plot_image(dataset,indx)
      n+=1

def check_balance(folders):
  F=[]
  for name in folders:
    path = name+'.pickle'
    print("loading %s dataset"%path)
    dataset =load_image(path)
    cs = check_size(dataset)
    print("size of sample = %d"%cs)
    F.append(cs)

  print("number of images in folders =",F)
  npf = np.array(F)
  print('Mean:', np.mean(npf))
  print('Standard deviation:', np.std(npf))


folders = train_folders # or train_folders
sample = SampleTrain # or SampleTrain

check_balance(folders)
sample_plot(folders,sample)





