import numpy as np
from PIL import Image
import glob
import random
import matplotlib.pyplot as plt
import os
from sklearn.cross_validation import train_test_split
import gzip
from six.moves import cPickle


random.seed(1992)
N = 24
starts = [(random.randint(0,128-N), random.randint(0,128-N)) for i in xrange(100)]

# Generate histograms
inputs = []
targets = []
for textureImage in glob.glob('OriginalBrodatz/*'):
    im = Image.open(textureImage)
    im.thumbnail((128,128), Image.ANTIALIAS)
    for x,y in starts:
        reg = im.crop((x,y,x+N,y+N))
        hist, bin_edges = np.histogram(np.asarray(reg), bins=16, range=(0,256), density=True)
        inputs.append(np.asarray(reg).flatten())
        targets.append(hist)

# Generate train, validate, and test date
xx_train, x_test, yy_train, y_test = train_test_split(inputs, targets, test_size=0.2, random_state=0)
x_train, x_validate, y_train, y_validate = train_test_split(xx_train, yy_train, test_size=0.2, random_state = 0)
trainSet = (x_train, y_train)
validSet = (x_validate, y_validate)
testSet = (x_test, y_test)

# pickle data 
toSave = (trainSet, validSet, testSet)
f = open('textureHistograms.pkl', 'wb')
cPickle.dump(toSave, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

# Print statistics
print 'Size of input: {}'.format(len(inputs[0]))
print 'Size of output: {}'.format(len(targets[0]))
print 'Number rows in x train: {}'.format(len(x_train))
print 'Number rows in y train: {}'.format(len(y_train))
print 'Number rows in x validate: {}'.format(len(x_validate))
print 'Number rows in y validate: {}'.format(len(y_validate))
print 'Number rows in x test: {}'.format(len(x_test))
print 'Number rows in y test: {}'.format(len(y_test))
