
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from Util import *

#Merge and prune the training data as needed. 
#The labels will be stored into a separate array of integers 0 through 9.
#Also create a validation dataset for hyperparameter tuning.

num_classes = 10
np.random.seed(133)

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.
pickle_file = 'notMNIST.pickle'  
train_size = 200000
valid_size = 10000
test_size = 10000

train_folders = ['notMNIST_large/A', 'notMNIST_large/B', 
'notMNIST_large/C', 'notMNIST_large/D', 'notMNIST_large/E', 
'notMNIST_large/F', 'notMNIST_large/G', 'notMNIST_large/H', 
'notMNIST_large/I', 'notMNIST_large/J']

test_folders =['notMNIST_small/A', 'notMNIST_small/B', 
'notMNIST_small/C', 'notMNIST_small/D', 'notMNIST_small/E', 
'notMNIST_small/F', 'notMNIST_small/G', 'notMNIST_small/H', 
'notMNIST_small/I', 'notMNIST_small/J']

train_datasets = ['notMNIST_large/A.pickle', 'notMNIST_large/B.pickle', 
'notMNIST_large/C.pickle', 'notMNIST_large/D.pickle', 'notMNIST_large/E.pickle', 
'notMNIST_large/F.pickle', 'notMNIST_large/G.pickle', 'notMNIST_large/H.pickle', 
'notMNIST_large/I.pickle', 'notMNIST_large/J.pickle']

test_datasets =['notMNIST_small/A.pickle', 'notMNIST_small/B.pickle', 
'notMNIST_small/C.pickle', 'notMNIST_small/D.pickle', 'notMNIST_small/E.pickle', 
'notMNIST_small/F.pickle', 'notMNIST_small/G.pickle', 'notMNIST_small/H.pickle', 
'notMNIST_small/I.pickle', 'notMNIST_small/J.pickle']


def make_arrays(nb_rows, img_size):
  """
  Prepare an array of images and an array of labels
  """

  print("make_array: dataset = np.ndarray(rows=%d, img_size=%d, img_size=%d)"%(
    nb_rows, img_size, img_size))

  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
  """
  Merge the datasets: in addition of training (train) data set
  create a validation (valid) dataset
  """

  
  print("merge_datasets: pickle files = ",pickle_files)
  num_classes = len(pickle_files)
  valid_dataset, valid_labels = make_arrays(valid_size, image_size)
  train_dataset, train_labels = make_arrays(train_size, image_size)
  vsize_per_class = valid_size // num_classes   #integer division (floor division)
  tsize_per_class = train_size // num_classes

  print("num_classes = %d"%num_classes)
  print("validation_size = %d"%valid_size)

  if valid_size > 0:
    print("validation_size per class = %d"%vsize_per_class)
    print("validation_labels (shape) = ",valid_labels.shape)
    print("training_size = %d"%train_size)
    print("training_size per class = %d"%tsize_per_class)
    print("training_labels (shape)= ",train_labels.shape)
  else:
    print("test_size = %d"%train_size)
    print("test_size per class = %d"%tsize_per_class)
    print("test_labels (shape)= ",train_labels.shape)
    
  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class

  print("start valid = %d, end valid =%d"%(start_v,end_v))
  print("start training = %d, end training =%d"%(start_t,end_t))

  wait()

  #enumerate(sequence, start=0)
  #Return an enumerate object. sequence must be a sequence, an iterator, 
  #or some other object which supports iteration. 
  #The next() method of the iterator returned by enumerate() 
  #returns a tuple containing a count (from start which defaults to 0) 
  #and the values obtained from iterating over sequence:

  #>>> seasons = ['Spring', 'Summer', 'Fall', 'Winter']
  #>>> list(enumerate(seasons))
  #[(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]

  for label, pickle_file in enumerate(pickle_files): 
    print("iterating over pickle files: label = %d"%label) 
    print("start valid = %d, end valid =%d"%(start_v,end_v))
    print("start training = %d, end training =%d"%(start_t,end_t)) 
    wait()

    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)  #training set corresponding to one letter
        # shuffle the letter set to have random validation and training set
        np.random.shuffle(letter_set)

        if valid_dataset is not None:
          valid_letter = letter_set[:vsize_per_class, :, :]  #choose a validation set
          valid_dataset[start_v:end_v, :, :] = valid_letter
          valid_labels[start_v:end_v] = label
          start_v += vsize_per_class
          end_v += vsize_per_class
                    
        train_letter = letter_set[vsize_per_class:end_l, :, :]
        train_dataset[start_t:end_t, :, :] = train_letter
        train_labels[start_t:end_t] = label
        start_t += tsize_per_class
        end_t += tsize_per_class
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise
    
  return valid_dataset, valid_labels, train_dataset, train_labels

def randomize(dataset, labels):
  """
  Randomize the data
  """
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels

def save_data(pickle_file):
  """
  pickle all the data into a large file
  """
  try:
    f = open(pickle_file, 'wb')
    save = {
      'train_dataset': train_dataset,
      'train_labels': train_labels,
      'valid_dataset': valid_dataset,
      'valid_labels': valid_labels,
      'test_dataset': test_dataset,
      'test_labels': test_labels,
      }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
  except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise
            


valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
  train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)

train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

print("after randomize")

print('Training labels:', train_labels)
print('Validation labels:', valid_labels)
print('Testing labels:', test_labels)

#save_data(pickle_file)
#statinfo = os.stat(pickle_file)
#print('Compressed pickle size:', statinfo.st_size)
