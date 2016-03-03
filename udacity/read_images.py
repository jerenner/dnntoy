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




###
def load_letter(folder, min_num_images):
  """Load the data for a single letter label (A,B,C...)"""

  # http://www.scipy-lectures.org/advanced/image_processing/
  image_files = os.listdir(folder)

  #An array object represents a multidimensional, 
  #homogeneous array of fixed-size items. 
  #An associated data-type object describes the format 
  #of each element in the array (its byte-order, 
  #how many bytes it occupies in memory, whether it is an integer, 
  #a floating point number, or something else, etc.)
  #These examples illustrate the low-level ndarray constructor. 

  #>>> np.ndarray(shape=(2,2), dtype=float, order='F')


  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32) #3D array of floats
  
  print(folder)

  image_index = 0
  for image in os.listdir(folder):
    image_file = os.path.join(folder, image)

    #ndimage.imread(image_file) reads the file into a nparray
    #np.astype(flotat) produces a copy of the array casted to float
    # every image is converted into a 3D array 
    #(image index, x, y) of floating point values, 
    #normalized to have approximately zero mean and standard deviation ~0.5 
    #to make training easier. To achieve that:
    # p = pixel_depth --> [0,255]
    # (p - 255/2.) --->[-255/2,255/2]  ---> 0 mean 
    # (p - 255/2.)/255 --->[-1/2,1/2]  ---> std 0.5

    try:
      image_data = (ndimage.imread(image_file).astype(float) - 
                    pixel_depth / 2) / pixel_depth

      #shape is a tuple with the array dimensions
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))

      dataset[image_index, :, :] = image_data #fills the image_data for each index
      image_index += 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
  num_images = image_index
  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))
    
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset


###        
def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  """
  pickle the data
  """
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)

      dataset = load_letter(folder, min_num_images_per_class)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)
  
  return dataset_names

train_datasets = maybe_pickle(train_folders, 45000) #45000/1800 
test_datasets = maybe_pickle(test_folders, 1800) #number of im per class