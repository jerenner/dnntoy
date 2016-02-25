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
pickle_file = 'notMNIST.pickle'
np.random.seed(143)

LETTER={0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J'}


def load_data(pickle_file):
  """
  pickle back the data as a dictionary 
  datadict = {
      'train_dataset': train_dataset,
      'train_labels': train_labels,
      'valid_dataset': valid_dataset,
      'valid_labels': valid_labels,
      'test_dataset': test_dataset,
      'test_labels': test_labels,
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

def plot_image(dataset,indx):
  image = dataset[indx]
  print("shape of image =",image.shape)
  img = (image + pixel_depth / 2) / pixel_depth
  plt.imshow(img)
  plt.show()

def label_image(dlabel,indx):
  lbl = dlabel[indx]
  print("label of image =%d: letter = %s"%(lbl,LETTER[lbl]))
  

def plot_dataset(dlabels,dataset,sample_size=5):
  print("number of labels =",dlabels.shape[0])
  sample =np.random.randint(0, 1000, size=sample_size)
  print("plotting random sample with indexes =",sample)

  for indx in sample:
    print("plotting picture with index =%d"%indx)
    label_image(dlabels,indx)
    plot_image(dataset,indx)

datadict = load_data(pickle_file)  
print("loaded datadict: keys =",datadict.keys()) 
train_dataset= datadict['train_dataset']
train_labels= datadict['train_labels'] 
valid_dataset= datadict['valid_dataset']
valid_labels= datadict['valid_labels']
test_dataset= datadict['test_dataset']
test_labels= datadict['test_labels'] 

print("plotting train dataset:")
plot_dataset(train_labels,train_dataset)
print("plotting valid dataset:")
plot_dataset(valid_labels,valid_dataset)
print("plotting test dataset:")
plot_dataset(test_labels,test_dataset)


