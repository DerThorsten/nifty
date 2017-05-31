from __future__ import absolute_import,print_function
from .. import graph

from skimage.feature import peak_local_max as __peak_local_max
from scipy.misc import imresize as __imresize
import numpy


def seededWatersheds(heightMap, seeds, method="node_weighted", acc="max"):

    hshape = heightMap.shape 
    sshape = seeds.shape 
    shape = sshape 
    ishape = [2*s -1 for s in shape]
    gridGraph = graph.gridGraph(shape)

    # node watershed
    if method == "node_weighted":
        assert hshape == sshape
        seg = graph.nodeWeightedWatershedsSegmentation(graph=gridGraph, seeds=seeds.ravel(),
                nodeWeights=heightMap.ravel())
        seg = seg.reshape(shape)

    if method == "edge_weighted":
        if acc != 'interpixel':
            assert hshape == sshape
            gridGraphEdgeStrength = gridGraph.imageToEdgeMap(heightMap, mode=acc)

        else:
            if(hshape == shape):
                iHeightMap = __imresize(heightMap, ishape, interp='bicubic')
                gridGraphEdgeStrength = gridGraph.imageToEdgeMap(iHeightMap, mode=acc)
            elif(hshape == ishape):
                gridGraphEdgeStrength = gridGraph.imageToEdgeMap(heightMap, mode=acc)
            else:
                raise RuntimeError("height map has wrong shape")

        seg = graph.edgeWeightedWatershedsSegmentation(graph=gridGraph, seeds=seeds.ravel(),
                    edgeWeights=gridGraphEdgeStrength)
        seg = seg.reshape(shape)


    return seg


def localMinima(image):
    """get the local minima of an image
    
    Get the local minima wrt a 4-neighborhood on an image.
    For a plateau, all pixels of this plateau are marked
    as minimum pixel.
    
    Args:
        image (numpy.ndarray): the input image
    
    Returns:
        (numpy.ndarray) : array which is 1 the minimum 0 elsewhere.

    """
    if image.ndim != 2:
        raise RuntimeError("localMinima is currently only implemented for 2D images")
    return localMaxima(-1.0*image)


def localMaxima(image):
    """get the local maxima of an image
    
    Get the local maxima wrt a 4-neighborhood on an image.
    For a plateau, all pixels of this plateau are marked
    as maximum pixel.
    
    Args:
        image (numpy.ndarray): the input image
    
    Returns:
        (numpy.ndarray) : array which is 1 the maximum 0 elsewhere.

    """
    if image.ndim != 2:
        raise RuntimeError("localMaxima is currently only implemented for 2D images")
    shape = image.shape
    lm = numpy.zeros(shape)
    coords = __peak_local_max(image)
    lm[coords[:,0], coords[:,1]] = 1
    return lm


def connectedComponents(labels, dense=True, ignoreBackground=False):
    shape = labels.shape
    gridGraph = graph.gridGraph(shape)

    ccLabels = graph.connectedComponentsFromNodeLabels(gridGraph, 
            labels.ravel(), dense=dense,
            ignoreBackground=bool(ignoreBackground))
    
    return ccLabels.reshape(shape)


def localMinimaSeeds(image):
    if image.ndim != 2:
        raise RuntimeError("localMinimaSeeds is currently only implemented for 2D images")

    lm = localMinima(image)
    cc = connectedComponents(lm, dense=True, ignoreBackground=True)
    return cc
    