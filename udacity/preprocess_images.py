"""
Traverses the notMNIST directories containing training and test images
In each directory creates a pickle file storing a numpy array of images
and a numpy array of hash strings, indexed by image number:

dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32) 

hashset = np.ndarray(shape=(len(image_files)),
                         dtype='|S16') #string array with the hash of images

"""
from UGen import *
nullh = '0000000000000000'


def load_letter(folder, num_images):
  """
  Loads the files found in folder (up to num_images)
  into a numpy array. 
  Computes the dHash of every image and loads into a np array
  """

  # http://www.scipy-lectures.org/advanced/image_processing/
  image_files = os.listdir(folder)

  print("folder = {0}: images in folder ={1}".format(folder,len(image_files)))
  wait()

  if len(image_files) < num_images:
    print("error: number of images requested ={0}, images in dir ={1}".format(num_images,
      image_files))
    sys.exit()

  dataset = np.ndarray(shape=(num_images, image_size, image_size),dtype=np.float32) 
  hashset = np.ndarray(shape=(num_images),dtype='|S16') 
  
  image_index = 0

  for image in image_files:
    if image_index == num_images:
      break

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

    #hash the file using dhash for later use 
    #(comparison of near identical images)
    shash =''
    try:
      xhash = imagehash.dhash(Image.open(image_file))
      shash = str(xhash)
    except IOError as e:
      print('Could not open image:', image_file, ':', e, 
        '-  skipping.')
      continue

    try:
      image_data = (ndimage.imread(image_file).astype(float) - 
                    pixel_depth / 2) / pixel_depth

      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))

      if shash == nullh:
        print("found null hash, skipping image")
      else:
        dataset[image_index, :, :] = image_data #fills data for each index
        hashset[image_index] = shash
      
        image_index += 1
      
    except IOError as e:
      print('Could not read:', image_file, ':', e, ' skipping.')
  
  if image_index < num_images:
    print("error: number of good images ={0}, images requested ={1}".format(image_index,
      num_images))
    sys.exit()  
   
  #dataset = dataset[0:num_images, :, :]
  #hashset = hashset[0:num_images]

   
  print("in folder ={0}: found = {1} good images".format(folder,num_images))
  print("dataset shape ={0}: hashset shape = {1} ".format(dataset.shape,hashset.shape))
  wait()
  return dataset, hashset


def pickle_datasets(data_folders, images_per_class, force=False):
  """
  1) Takes a list of folders
  2) Traverses folders and load letters for each folder
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

      dataset, hashset = load_letter(folder, images_per_class)
      
      pickle_data(set_filename,dataset, hashset)

      
  
  return dataset_names

def pickle_data(pickle_file,dataset, hashset):
  """
  pickle all the data into a large file
  """
  try:
    f = open(pickle_file, 'wb')
    save = {
      'letter_dataset': dataset,
      'letter_hashset': hashset
      }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
  except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise




def demo(folder,number_of_images=5):
  """
  1) Scans specified folder
  2) Pickles specified number of images
  3) Reads back the images and hashes and displays them
  """
  
  print("scanning folder = {}".format(folder))
  print("reading {} images".format(number_of_images))

  dataset, hashset = load_letter(folder, number_of_images)
  print("dataset", dataset)
  wait()
  print("hashset", hashset)
  wait()

  print('dataset shape ={0}, hashset shape ={1}:'.format(
    dataset.shape,hashset.shape))
    
  print('Mean of dataset ={0:.2f}'.format(
    np.mean(dataset)))
  print('Standard deviation ={0:.2f}'.format(
    np.std(dataset)))
  pickle_file = '{}.pickle'.format(folder)
  print("Write into pickle file = {}".format(pickle_file))
  pickle_data(pickle_file,dataset, hashset)

  print("Read from pickle file = {}".format(pickle_file))
  dataset2, hashset2 =load_data(pickle_file)

  for indx in range(number_of_images):
    plot_image(dataset2,indx)
    compare_hash(hashset,hashset2,indx)



if __name__ == '__main__':

  import sys, os
  def usage():
    sys.stderr.write("""SYNOPSIS: %s [demo|test|train|dtest|dtrain] 

Preprocess the notMNIST data.

Method: 
  demo: run over a few images, display images and hashes
  test: pickles test data
  train: pickles train data
  all: pickles test and train data
  dtest: display images for test files
  dtrain: display images for train files

JJGC, adapted and expanded from Udacity scripts
""" % sys.argv[0])
    sys.exit(1)
    
  action = sys.argv[1] if len(sys.argv) > 1 else usage()
  if action == 'demo':
    print("demo in folder ={0}, number of images ={1}".format(
      'notMNIST_small/A',10))
    demo('notMNIST_small/A',number_of_images=10)
  elif action == 'test':

    test_folders =['notMNIST_small/A', 'notMNIST_small/B', 
        'notMNIST_small/C', 'notMNIST_small/D', 'notMNIST_small/E', 
        'notMNIST_small/F', 'notMNIST_small/G', 'notMNIST_small/H', 
        'notMNIST_small/I', 'notMNIST_small/J']
    
    test_images = 1600
    print("pickle test dataset: test folders ={0}: images ={1}".
      format(test_folders, test_images))

    test_dataset_names =pickle_datasets(test_folders, test_images, 
                                      force=True)
  elif action == 'train':
    train_folders = ['notMNIST_large/A', 'notMNIST_large/B', 
    'notMNIST_large/C', 'notMNIST_large/D', 'notMNIST_large/E', 
    'notMNIST_large/F', 'notMNIST_large/G', 'notMNIST_large/H', 
    'notMNIST_large/I', 'notMNIST_large/J']

    train_images = 45000

    print("pickle train dataset: test folders ={0}: images ={1}".
      format(train_folders, train_images))

    train_dataset_names =pickle_datasets(train_folders, train_images, 
                                      force=True)
  
  elif action == 'dtest':
    print("dtest")
    print("display test datasets")
    test_datasets =['notMNIST_small/A.pickle', 'notMNIST_small/B.pickle', 
    'notMNIST_small/C.pickle', 'notMNIST_small/D.pickle', 
    'notMNIST_small/E.pickle', 
    'notMNIST_small/F.pickle', 'notMNIST_small/G.pickle', 
    'notMNIST_small/H.pickle', 
    'notMNIST_small/I.pickle', 'notMNIST_small/J.pickle']

    for pickle_file in test_datasets:
      sample =np.random.randint(0, 100, size=3)
      display(pickle_file,sample)

  elif action == 'dtrain':
    print("dtrain")
    print("display train datasets")
    train_datasets = ['notMNIST_large/A.pickle', 'notMNIST_large/B.pickle', 
  'notMNIST_large/C.pickle', 'notMNIST_large/D.pickle', 
  'notMNIST_large/E.pickle', 
  'notMNIST_large/F.pickle', 'notMNIST_large/G.pickle', 
  'notMNIST_large/H.pickle', 
  'notMNIST_large/I.pickle', 'notMNIST_large/J.pickle']

    for pickle_file in train_datasets:
      sample =np.random.randint(0, 100, size=3)
      display(pickle_file,sample)
  else:
    usage()

  