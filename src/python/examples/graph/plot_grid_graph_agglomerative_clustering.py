"""
Agglomerative clustering on grid graph
========================================

Compare edge weighted watersheds
and node weighted on a grid graph.
"""

####################################
# sphinx_gallery_thumbnail_number = 1
from __future__ import print_function

import nifty.graph
import nifty.graph.agglo
import skimage.data 
import skimage.segmentation 
import vigra
import matplotlib
import pylab
import numpy

# increase default figure size
a,b = pylab.rcParams['figure.figsize']
pylab.rcParams['figure.figsize'] = 2.0*a, 2.0*b


####################################
# load some image
img = skimage.data.coins().astype('float32')
shape = img.shape[0:2]
#plot the image
# pylab.imshow(img/255)
# pylab.show()





#########################################
# grid graph
gridGraph = nifty.graph.undirectedGridGraph(shape)



#########################################
# the edge weights for a grid graph work
# best if we use interpixel weights.
# Therefore we need to resample the
# image.
# On the resampled image, an edge indicator 
# is computed
interpixelShape = [2*s-1 for s in shape]

# to vigra
tags = ['xy','xyz'][img.ndim-2]
vigraImg = vigra.taggedView(img, tags)
imgBig = vigra.sampling.resize(vigraImg, interpixelShape)
edgeStrength = vigra.filters.gaussianGradientMagnitude(imgBig, 2.0)
edgeStrength = edgeStrength.squeeze()
edgeStrength = numpy.array(edgeStrength)
gridGraphEdgeStrength = gridGraph.imageToEdgeMap(edgeStrength, mode='interpixel')



#########################################
#run agglomerative clustering
edgeSizes = numpy.ones(gridGraph.edgeIdUpperBound +1)
nodeSizes = numpy.ones(gridGraph.nodeIdUpperBound +1)
clusterPolicy = nifty.graph.agglo.edgeWeightedClusterPolicy(
    graph=gridGraph, edgeIndicators=gridGraphEdgeStrength,
    edgeSizes=edgeSizes, nodeSizes=nodeSizes,
    numberOfNodesStop=25, sizeRegularizer=0.35)
agglomerativeClustering = nifty.graph.agglo.agglomerativeClustering(clusterPolicy) 
agglomerativeClustering.run()
seg = agglomerativeClustering.result()
seg = seg.reshape(shape)



#########################################
# plot results
b_img = skimage.segmentation.mark_boundaries(img/255, 
        seg.astype('uint32'), mode='inner', color=(0.1,0.1,0.2))
pylab.imshow(b_img)
pylab.title('Segmentation')
pylab.show()