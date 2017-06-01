from __future__ import absolute_import,print_function
from .. import graph

from skimage.feature import peak_local_max as __peak_local_max
from skimage.segmentation import mark_boundaries as __mark_boundaries
from scipy.misc import imresize as __imresize
from scipy.ndimage import zoom as __zoom

import numpy


def seededWatersheds(heightMap, seeds, method="node_weighted", acc="max"):
    """Seeded watersheds segmentation
    
    Get a segmentation via seeded watersheds.
    This is a high level wrapper around 
    :func:`nifty.graph.nodeWeightedWatershedsSegmentation`
    and :func:`nifty.graph.nodeWeightedWatershedsSegmentation`.

    
    Args:
        heightMap (numpy.ndarray) : height / evaluation map  
        seeds (numpy.ndarray) : Seeds as non zero elements in the array
        method (str): Algorithm type can be:

            *   "node_weighted": ordinary node weighted watershed
            *   "edge_weighted": edge weighted watershed (minimum spanning tree)
        
            (default: {"node_weighted"})

        acc (str): If method is "edge_weighted", one needs to specify how
            to convert the heightMap into an edgeMap.
            This parameter specificities this method.
            Allow values are:

            *   'min' : Take the minimum value of the endpoints of the edge 
            *   'max' : Take the minimum value of the endpoints of the edge 
            *   'sum' : Take the sum  of the values of the endpoints of the edge 
            *   'prod' : Take the product of the values of the endpoints of the edge 
            *   'interpixel' : Take the value of the image at the interpixel
                coordinate in between the two endpoints.
                To do this the image is resampled to have shape :math: `2 \cdot shape -1 `

            (default: {"max"})
    
    Returns:
        numpy.ndarray : the segmentation
    
    Raises:
        RuntimeError: [description]
    """
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

    elif method == "edge_weighted":
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
    """get connected components of a label image
    
    Get connected components of an image w.r.t.
    a 4-neighborhood .      
    This is a high level wrapper for
        :func:`nifty.graph.connectedComponentsFromNodeLabels` 

    Args:
        labels (numpy.ndarray): 
        dense (bool): should the return labeling be dense (default: {True})
        ignoreBackground (bool): should values of zero be excluded (default: {False})
    
    Returns:
        [description]
        [type]
    """
    shape = labels.shape
    gridGraph = graph.gridGraph(shape)

    ccLabels = graph.connectedComponentsFromNodeLabels(gridGraph, 
            labels.ravel(), dense=dense,
            ignoreBackground=bool(ignoreBackground))
    
    return ccLabels.reshape(shape)


def localMinimaSeeds(image):
    """Get seed from local minima
    
    Get seeds by running connected components 
    on the local minima.
    This is a high level wrapper around
    :func:`nifty.segmentation.localMinima` 
    and :func:`nifty.segmentation.connectedComponents` 
    
    Args:
        image: [description]
    
    Returns:
        [description]
        [type]
    
    Raises:
        RuntimeError: [description]
    """
    if image.ndim != 2:
        raise RuntimeError("localMinimaSeeds is currently only implemented for 2D images")

    lm = localMinima(image)
    cc = connectedComponents(lm, dense=True, ignoreBackground=True)
    return cc
    


def markBoundaries(image, segmentation, color=None):
    """Mark the boundaries in an image
    
    Mark boundaries in an image.

    Warning:

        The returned image shape is twice as large
        as the input.
    
    Args:
        image:  the input image 
        segmentation:  the segmentation
        color (tuple) : the edge color(default: {(0,0,0)})
    
    Returns:
        (numpy.ndarray) : image with marked boundaries. Note that
            result image has twice as large shape as the input.
    """
    if color is None:
        color = (0,0,0)

    shape = segmentation.shape
    img2 =   __imresize(image, [2*s for s in shape])#, interp='nearest')
    #img2 = __zoom(segmentation.astype('float32'), 2, order=1)
    seg2 = __zoom(segmentation.astype('float32'), 2, order=0)
    seg2 = seg2.astype('uint32')

    return __mark_boundaries(img2, seg2.astype('uint32'), color=color)

