import numpy as np
from PIL import Image
import glob
import random
import matplotlib.pyplot as plt
import os

outputDir = '/Users/mnajarian/UNC_GoogleDrive/Spring2017/777/project/histograms'

random.seed(1992)
N = 24
starts = [(random.randint(0,128-N), random.randint(0,128-N)) for i in xrange(50)]

# Generate histograms for presentation

for textureImage in glob.glob('OriginalBrodatz/*'):
    figName = os.path.splitext(os.path.basename(textureImage))[0]
    im = Image.open(textureImage)
    im.thumbnail((128,128), Image.ANTIALIAS)
    for x,y in starts:
        reg = im.crop((x,y,x+N,y+N))
        hist, bin_edges = np.histogram(np.asarray(reg), bins=16, range=(0,256), density=True)
        plt.plot(bin_edges[:-1], hist)
    plt.title('Histogram {}'.format(figName))
    plt.xlabel('Pixel value')
    plt.ylabel('Normalized frequency')
    plt.savefig('{}/{}.png'.format(outputDir, figName))
    plt.clf()
    print figName