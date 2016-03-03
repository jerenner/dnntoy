from UGen import *


train_datasets = ['notMNIST_large/A.pickle', 'notMNIST_large/B.pickle', 
'notMNIST_large/C.pickle', 'notMNIST_large/D.pickle', 'notMNIST_large/E.pickle', 
'notMNIST_large/F.pickle', 'notMNIST_large/G.pickle', 'notMNIST_large/H.pickle', 
'notMNIST_large/I.pickle', 'notMNIST_large/J.pickle']

test_datasets =['notMNIST_small/A.pickle', 'notMNIST_small/B.pickle', 
'notMNIST_small/C.pickle', 'notMNIST_small/D.pickle', 'notMNIST_small/E.pickle', 
'notMNIST_small/F.pickle', 'notMNIST_small/G.pickle', 'notMNIST_small/H.pickle', 
'notMNIST_small/I.pickle', 'notMNIST_small/J.pickle']

if __name__ == '__main__':
  import sys, os
  def usage():
    sys.stderr.write("""
      SYNOPSIS: %s set, letter, imin, imax, number_of_images

      Displays images and print hashes reading letters pickle files.

      arguments: 
      set: tragin of test
      letter: A to J
      imin and imax define the range
      number_of_images: number of images in range above
      
      if no arguments given, use defaults

      set = train
      letter = A
      imin = 0
      imax = 100
      number_of_images = 5
      
  
      JJGC, adapted and expanded from Udacity scripts
      """ % sys.argv[0])
    sys.exit(1)

  lset = 'train'
  letter = 'A'
  imin = 0
  imax = 100
  number_of_images = 5 

  print("len(sys.argv) ={0}".format(len(sys.argv)))
  print("sys.argv ={0}".format(sys.argv))
  

  if len(sys.argv) == 1 : 
    pass 
  elif len(sys.argv) == 2 :
    lset =  sys.argv[1]
  elif len(sys.argv) == 3 :
    lset =  sys.argv[1]
    letter = sys.argv[2]
  elif len(sys.argv) == 4 :
    lset =  sys.argv[1]
    letter = sys.argv[2]
    imin = int(sys.argv[3])
  elif len(sys.argv) == 5 :
    lset =  sys.argv[1]
    letter = sys.argv[2]
    imin = int(sys.argv[3])
    imax = int(sys.argv[4])
  elif len(sys.argv) == 6 :
    lset =  sys.argv[1]
    letter =  sys.argv[2]
    imin = int(sys.argv[3])
    imax = int(sys.argv[4])
    number_of_images =int(sys.argv[5])
  else:
    usage()

  dir_name = 'notMNIST_large'
  if lset == 'test':
    dir_name = 'notMNIST_small'

  pickle_file ='{0}/{1}.pickle'.format(dir_name,letter)
  sample =np.random.randint(imin, imax, size=number_of_images)

  print("""
    Displaying {0} images 
    in the range between {1} and {2}
    from file {3}
    set of images displayed ={4}
    
    """.format(number_of_images,imin,imax,pickle_file,sample))

  display(pickle_file, sample)


