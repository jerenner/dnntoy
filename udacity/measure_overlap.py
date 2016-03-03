from UGen import *


LETTER={0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J'}

if __name__ == '__main__':
  import sys, os
  def usage():
    sys.stderr.write("""
      SYNOPSIS: %s train_size, valid_size, test_size, pickle_file

      measures overlaps in  the notMNIST data 

      arguments: 
      train_size, valid_size, test_size: size of the training, validation and test size
      to be scanned 
      pickle_file: name of pickle file
      if no arguments given, use defaults

      
  
      JJGC, adapted and expanded from Udacity scripts
      """ % sys.argv[0])
    sys.exit(1)

  train_size = 200000
  valid_size = 10000
  test_size = 10000 
  pickle_file = 'notMNIST_merged_train_{0}_valid_{1}_test_{2}.pickle'.format(
    train_size,valid_size,test_size)

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
    pickle_file = sys.argv[4]
  else:
    usage()

  datadict = load_merged_data(pickle_file)  
  print("loaded datadict: keys =",datadict.keys()) 

  train_dataset= datadict['train_dataset']
  train_labels= datadict['train_labels']
  train_hashset= datadict['train_hashset'] 

  valid_dataset= datadict['valid_dataset']
  valid_labels= datadict['valid_labels']
  valid_hashset= datadict['valid_hashset']

  test_dataset= datadict['test_dataset']
  test_labels= datadict['test_labels'] 
  test_hashset= datadict['test_hashset']


  overlaps =0
  overlap_label={}

  print("length of testset = {}".format(test_hashset.shape[0]))
  print("length of trainset = {}".format(train_hashset.shape[0]))
  print("length of validset = {}".format(valid_hashset.shape[0]))

  print("testing with hashes")
  for itest in range(test_hashset.shape[0]):
    oit = 0
    #print("itest = {0}, label ={1}, hash ={2}".format(itest,
      #test_labels[itest],test_hashset[itest]))
    
    testhset = test_hashset[itest]
  # check if this guy is present in the train dataset 
    for itrain in range(train_hashset.shape[0]):
    
      trainhset = train_hashset[itrain]
      if test_hashset[itest] == train_hashset[itrain]:
        print("found overlap, itest ={0} itrain ={1} hash = {2}".format(itest,itrain,
          test_hashset[itest]))
        overlaps+=1
        oit+=1
    overlap_label[test_labels[itest]]=oit
  
  print("hash method")
  print("found {0} overlaps between train and test:".format(overlaps))
  print("overlaps per class = {}".format(overlap_label))
  # for label in range(10):
  #   print("For label index {0} letter {1} overlaps ={2}:".format(
  #     label,LETTER[label],overlap_label[label]))


  print("testing with direct image comparison")
  overlap_label={}
  overlaps =0
  for itest in range(test_dataset.shape[0]):
    oit = 0
    print("itest = {0}, label ={1}, hash ={2}".format(itest,
      test_labels[itest],test_hashset[itest]))

    testdset = test_dataset[itest]
    # check if this guy is present in the train dataset 
    
    for itrain in range(train_dataset.shape[0]):
      traindset = train_dataset[itrain]

      if np.array_equal(testdset,traindset):
        print("found overlap, itest ={0} itrain ={1} hashtest = {2} hashtrain = {3}".format(
          itest,itrain,test_hashset[itest],train_hashset[itrain]))
        print("compare hashes = {}".format(test_hashset[itest]==train_hashset[itrain]))
        plot_image(test_dataset,itest)
        plot_image(train_dataset,itrain)
      
        overlaps+=1
        oit+=1
        
    overlap_label[test_labels[itest]]=oit

  print("direct comparison method")
  print("found {0} overlaps between train and test:".format(overlaps))
  print("overlaps per class = {}".format(overlap_label))


  

#testdset = test_dataset[0]

# image = test_dataset[0]
# print("shape of image =",image.shape)
# img = (image + pixel_depth / 2) / pixel_depth
# plt.imshow(img)
# plt.show()
# hash = imagehash.dhash(Image.open(img))
# print(" hash =",hash)



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

