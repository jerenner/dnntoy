from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from os.path import isfile, join

from Util import *

import matplotlib.image as mpimg
from scipy import ndimage
from six.moves import cPickle as pickle
import imagehash

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

datadict = load_data(pickle_file)  
print("loaded datadict: keys =",datadict.keys()) 
train_dataset= datadict['train_dataset']
train_labels= datadict['train_labels'] 
valid_dataset= datadict['valid_dataset']
valid_labels= datadict['valid_labels']
test_dataset= datadict['test_dataset']
test_labels= datadict['test_labels'] 

testdset = test_dataset[0]

image = test_dataset[0]
print("shape of image =",image.shape)
img = (image + pixel_depth / 2) / pixel_depth
plt.imshow(img)
plt.show()
hash = imagehash.dhash(Image.open(img))
print(" hash =",hash)
# ott =0
# ott_labels=[]
# for itest in range(test_dataset.shape[0]):
#   oit = 0
#   print("itest = %d"%itest)
#   print("labels =",ott_labels)

#   testdset = test_dataset[itest]
#   # check if this guy is present in the train dataset 
#   for itrain in range(train_dataset.shape[0]):
#     traindset = train_dataset[itrain]
#     if np.array_equal(testdset,traindset):
#       print("found overlap, itest =%d, itrain = %d"%(itest,itrain))
#       ott+=1
#       oit+=1
#   ott_labels.append((itest,oit))

# print("found %d overlaps between train and test:"%(ott))
# print("overlaps between train and test->",ott_labels

