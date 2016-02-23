# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os

from os.path import isfile, join

from Util import *

import matplotlib.image as mpimg





def get_filepaths(directory):
    """
    This function will generate the file names in a directory 
    tree by walking the tree either top-down or bottom-up. For each 
    directory in the tree rooted at directory top (including top itself), 
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths  # Self-explanatory.

def list_files(mypath):
	
	files = [f for f in os.listdir(mypath) if isfile(join(mypath, f))]
	return files


def list_files_startwith(mypath, startwith):
	
	files = [f for f in os.listdir(mypath) if isfile(join(mypath, f))]
	sfiles =[f for f in files if f.startswith(startwith) == True]
	
	return sfiles

dirnames =["A",	"B","C","D","E","F","G","H","I","J"] 
notMINST =["notMNIST_large","notMNIST_small"]

mpath = os.getcwd()+"/"+notMINST[0]+"/"+dirnames[1]
print("current directory = %s"%mpath)


files = list_files_startwith(mpath,"a2F")
print("files in %s = "%mpath,files)

listOfImageNames = [mpath+"/"+f for f in files]

print("listOfImageNames =  ",listOfImageNames)

for imageName in listOfImageNames:
	print("display image =  ",imageName)
	img = mpimg.imread(imageName)
	plt.imshow(img)
	plt.show()
	#display(Image(filename=imageName))
	#wait()

