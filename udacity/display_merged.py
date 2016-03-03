from UGen import *


if __name__ == '__main__':
  import sys, os
  def usage():
    sys.stderr.write("""
      SYNOPSIS: %s file, dset, imin, imax, number_of_images

      Displays images and print hashes reading merged pickle file.

      arguments: 
      file: pickle file with merged data sets
      dset: train, valid, test
      imin and imax define the range
      number_of_images: number of images in range above
      
      if no arguments given, use defaults
      file = notMNIST_merged_train_2000_valid_100_test_100.pickle
      dset = train
      
      imin = 0
      imax = 100
      number_of_images = 5
      
  
      JJGC, adapted and expanded from Udacity scripts
      """ % sys.argv[0])
    sys.exit(1)

  dset = 'train'
  file = 'notMNIST_merged_train_2000_valid_100_test_100.pickle'
  imin = 0
  imax = 100
  number_of_images = 5 


  if len(sys.argv) == 1 : 
    pass 
  elif len(sys.argv) == 2 :
    file = sys.argv[1]
  elif len(sys.argv) == 3 :
    file = sys.argv[1]
    dset =  sys.argv[2]
  elif len(sys.argv) == 4 :
    file = sys.argv[1]
    dset =  sys.argv[2]
    imin = int(sys.argv[3])
  elif len(sys.argv) == 5 :
    file = sys.argv[1]
    dset =  sys.argv[2]
    imin = int(sys.argv[3])
    imax = int(sys.argv[4])
  elif len(sys.argv) == 6 :
    file = sys.argv[1]
    dset =  sys.argv[2]
    imin = int(sys.argv[3])
    imax = int(sys.argv[4])
    number_of_images =int(sys.argv[5])
  else:
    usage()

  sample =np.random.randint(imin, imax, size=number_of_images)

  print("""
    Displaying {0} images 
    from set {1}
    in the range between {2} and {3}
    from file {4}
    set of images displayed ={5}
    
    """.format(number_of_images,dset,imin,imax,file,sample))

  display_merged(file, dset, sample)


