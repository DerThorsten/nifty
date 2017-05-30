"""
Grid Graph Edge Weighted Watersheds
====================================
Edge Weighted Watersheds on a Undirected Grid Graph

Warning:

    This function is still somewhat experimental
"""

####################################
# load modules
# and do some minor setup
from __future__ import print_function

import nifty.graph
import skimage.data 
import skimage.segmentation 
import vigra
import matplotlib
import pylab
import numpy

# increase default figure size
a,b = pylab.rcParams['figure.figsize']
pylab.rcParams['figure.figsize'] = 1.5*a, 1.5*b


####################################
# load some image
img = skimage.data.astronaut().astype('float32')
shape = img.shape[0:2]

#plot the image
pylab.imshow(img/255)
pylab.show()

###################################################
# get some edge indicator to get seeds from
taggedImg = vigra.taggedView(img,'xyc')
edgeStrength = vigra.filters.structureTensorEigenvalues(taggedImg, 1.0, 4.0)[:,:,0]
edgeStrength = edgeStrength.squeeze()
pylab.imshow(edgeStrength)
pylab.show()

###################################################
# get seeds via local minima
seeds = vigra.analysis.localMinima(edgeStrength)
seeds = vigra.analysis.labelImageWithBackground(seeds)

# plot seeds
cmap =  numpy.random.rand ( seeds.max()+1,3)
cmap[0,:] = 0
cmap = matplotlib.colors.ListedColormap ( cmap)
pylab.imshow(seeds, cmap=cmap)
pylab.show()

#########################################
# grid graph
gridGraph = nifty.graph.undirectedGridGraph(shape)



#########################################
# edgeStrength
edgeStrength = vigra.filters.gaussianGradientMagnitude(vigra.taggedView(img,'xyc'), 1.0)
edgeStrength = edgeStrength.squeeze()
pylab.imshow(edgeStrength)
pylab.show()


#########################################
# convert image to grid graph edge map
gridGraphEdgeStrength = gridGraph.imageToEdgeMap(edgeStrength, mode='sum')
numpy.random.permutation(gridGraphEdgeStrength)


#########################################
# run the actual algorithm
overseg = nifty.graph.edgeWeightedWatershedSegmentation(graph=gridGraph, seeds=seeds.ravel(),
    edgeWeights=gridGraphEdgeStrength.ravel())
overseg = overseg.reshape(shape)


#########################################
# result
b_img = skimage.segmentation.mark_boundaries(img/255, 
        overseg.astype('uint32'), mode='inner', color=(0,0,0))
pylab.imshow(b_img)
pylab.show()