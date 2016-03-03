
from UGen import *

#Merge and prune the training data as needed. 
#The labels will be stored into a separate array of integers 0 through 9.
#Also create a validation dataset for hyperparameter tuning.

num_classes = 10

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.



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
  Prepare an array of images, and array of hashed and an array of labels
  """

  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
    hashset = np.ndarray(nb_rows,dtype='|S16') 
  else:
    dataset, labels, hashset = None, None, None
  return dataset, labels, hashset

def merge_datasets(pickle_files, train_size, valid_size=0):
  """
  Merge the datasets: in addition of training (train) data set
  create a validation (valid) dataset
  """

  print("merge_datasets: pickle files = ",pickle_files)
  num_classes = len(pickle_files)
  valid_dataset, valid_labels, valid_hashset = make_arrays(valid_size, image_size)
  train_dataset, train_labels, train_hashset  = make_arrays(train_size, image_size)
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

  for label, pickle_file in enumerate(pickle_files):

    print("iterating over pickle files: label = %d"%label) 
    print("start valid = %d, end valid =%d"%(start_v,end_v))
    print("start training = %d, end training =%d"%(start_t,end_t)) 
   
    try:
      letter_set, hash_set = load_data(pickle_file)
      #image and has seta corresponding to one letter
      # shuffle the letter set to have random validation and training set
      
      permutation = np.random.permutation(letter_set.shape[0])
      
      shuffled_letterset = letter_set[permutation,:,:]
      shuffled_hashset = hash_set[permutation]

      # print("permutacion ={}".format(permutation))
      # print("hash_set[permutation[0]] ={}".format(hash_set[permutation[0]]))
      # print("shuffled_hashset[0] ={}".format(shuffled_hashset[0]))
      # wait()
      #np.random.shuffle(letter_set)

      if valid_dataset is not None:
        #valid_letter = letter_set[:vsize_per_class, :, :]  #choose a validation set
        valid_dataset[start_v:end_v, :, :] = shuffled_letterset[:vsize_per_class, :, :]
        #valid_hash = hash_set[:vsize_per_class]  #choose a validation set
        valid_hashset[start_v:end_v] = shuffled_hashset[:vsize_per_class]
        valid_labels[start_v:end_v] = label

        # print ("valid_hashset[0] ={}".format(valid_hashset[0]))
        # wait()

        start_v += vsize_per_class
        end_v += vsize_per_class
                    
      #train_letter = letter_set[vsize_per_class:end_l, :, :]
      train_dataset[start_t:end_t, :, :] = shuffled_letterset[vsize_per_class:end_l, :, :]
      #train_hash = hash_set[vsize_per_class:end_l]
      train_hashset[start_t:end_t] = shuffled_hashset[vsize_per_class:end_l]
      train_labels[start_t:end_t] = label

      #print ("train_hashset[10] ={}".format(train_hashset[10]))
      #wait()

      start_t += tsize_per_class
      end_t += tsize_per_class

    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise
    
  return (valid_dataset, valid_labels, valid_hashset),(train_dataset, 
    train_labels, train_hashset)

def randomize(dataset, labels, hashset):
  """
  Randomize the data
  """
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_hashset = hashset[permutation]
  shuffled_labels = labels[permutation]
  return (shuffled_dataset, shuffled_labels, shuffled_hashset)


def save_merged_data(pickle_file, train, valid, test):
  """
  pickle all the merged data into a large file. 
  train, valid and test are tuples containing dataset, label and hashset
  """
  train_dataset = train[0]
  train_hashset = train[2]
  train_labels = train[1]

  valid_dataset = valid[0]
  valid_hashset = valid[2]
  valid_labels = valid[1]

  test_dataset = test[0]
  test_hashset = test[2]
  test_labels = test[1]
  print("""
    train labels = {0} hashset ={1}
    valid labels = {2} hashset ={3}
    test labels = {4} hashset ={5}
    """.format(train_labels[0:10],train_hashset[0:10],
              valid_labels[0:10],valid_hashset[0:10],
              test_labels[0:10],test_hashset[0:10]
      ))
  try:
    f = open(pickle_file, 'wb')
    save = {
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
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
  except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise
            

if __name__ == '__main__':
  import sys, os
  def usage():
    sys.stderr.write("""
      SYNOPSIS: %s train_size, valid_size, test_size, random_seed

      Merges the notMNIST data in a large file.

      arguments: 
      train_size, valid_size, test_size: size of the training, validation and test size. 
      random_seed: changes the random seed for data shuffling
      if no arguments given, use defaults

      np.random.seed(133)
      pickle_file = 'notMNIST_merged.pickle'  

      train_size = 2000
      valid_size = 100
      test_size = 100
  
      JJGC, adapted and expanded from Udacity scripts
      """ % sys.argv[0])
    sys.exit(1)

  seed = 133
  train_size = 2000
  valid_size = 100
  test_size = 100 
      
  if len(sys.argv) == 1 : 
    pass 
  elif len(sys.argv) == 2 :
    train_size = int(sys.argv[1])
  elif len(sys.argv) == 3 :
    train_size = int(sys.argv[1])
    valid_size = int(sys.argv[2])
  elif len(sys.argv) == 4 :
    train_size = int(sys.argv[1])
    valid_size = int(sys.argv[2])
    test_size = int(sys.argv[3])
  elif len(sys.argv) == 5 :
    train_size = int(sys.argv[1])
    valid_size = int(sys.argv[2])
    test_size = int(sys.argv[3])
    seed = int(sys.argv[4])
  else:
    usage()

  pickle_file = 'notMNIST_merged_train_{1}_valid_{2}_test_{3}.pickle'.format(
    seed,train_size,valid_size,test_size)

  print("""Merging with  values
                train_size = {0}
                valid_size = {1}
                test_size = {2}
                random seed ={3}
                pickle_file ={4}
               """.format(train_size,valid_size,test_size, seed, pickle_file)
        )

  np.random.seed(seed)
    
  valid, train = merge_datasets(train_datasets, train_size, valid_size)
  valid_dataset = valid[0] 
  valid_labels = valid[1] 
  valid_hash = valid[2]

  train_dataset = train[0] 
  train_labels = train[1] 
  train_hash = train[2]

  _, test = merge_datasets(test_datasets, test_size)

  test_dataset = test[0] 
  test_labels = test[1] 
  test_hash = test[2]

  print('Training:', train_dataset.shape, train_labels.shape, train_hash.shape)
  print('Validation:', valid_dataset.shape, valid_labels.shape, valid_hash.shape)
  print('Testing:', test_dataset.shape, test_labels.shape, test_hash.shape)

  train = randomize(train_dataset, train_labels, train_hash)  
  test = randomize(test_dataset, test_labels, test_hash)
  valid = randomize(valid_dataset, valid_labels, valid_hash)
  save_merged_data(pickle_file, train, valid, test)


  statinfo = os.stat(pickle_file)
  print('Compressed pickle size:', statinfo.st_size)
