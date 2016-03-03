"Softmax."
from Util import *

import numpy as np
import matplotlib.pyplot as plt


def softmax(x):
	"""Compute softmax values for each sets of scores in x.
	Notice that we sum according to the axis=0. This is important when 
	sending multi-dim nparrays to softmax
	"""
	
 	return np.exp(x)/np.sum(np.exp(x), axis=0)


# Plot softmax curves. Prepare a set of scores of the form [x,1,0.2]
# where x varies in a range between small score (-2) and large score (6)

x = np.arange(-2.0, 6.0, 0.1)  #this prepares the first component of the vector of scores
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

#This stacks the scores as a vector [[x],[1],[0.2]], where [x] varies across the range
# and [1] and [0.2] repeat. that is [[-2.,-1.9...,5.8,5.9],[1,1,...1],[0.2,0.2...0.2]]
#The elements of the transpose of this vector are the set of scores that we want to softwax

#scores.T[0] = [-2.,1,0.2] and so on

print "vector of scores"
for i in range(0,3):
	print "scores[%d]= "%i,scores[i] 
wait()

print "vector of scores transposed"
for i in range(0,len(x)):
	print "scores.T[%d]= "%i,scores.T[i] 
wait()


print "sum of scores across axis 0"
print np.sum(scores, axis=0)
wait()
print "sum of scores across axis 1"
print np.sum(scores, axis=1)
wait()

print "plotting the three sets: blue is y[0]=x, red is y[1]=1, green is y[2]=0.2"
plt.plot(x, scores.T, linewidth=2)
plt.show()

print "plotting the softmax of scores"
plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()