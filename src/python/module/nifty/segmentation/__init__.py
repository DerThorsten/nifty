from __future__ import absolute_import,print_function
from .. import graph
from .. import filters

from skimage.feature import peak_local_max as __peak_local_max
import skimage.segmentation
from scipy.misc import imresize as __imresize
from scipy.ndimage import zoom as __zoom
import scipy.ndimage
from matplotlib.colors import ListedColormap as __ListedColormap
import numpy



def slic(image, nSegments, components):
    """ same as skimage.segmentation.slic """
    return skimage.segmentation.slic(image, n_segments=nSegments,
        compactness=compactness)


def seededWatersheds(heightMap, seeds=None, method="node_weighted", acc="max"):
    """Seeded watersheds segmentation
    
    Get a segmentation via seeded watersheds.
    This is a high level wrapper around 
    :func:`nifty.graph.nodeWeightedWatershedsSegmentation`
    and :func:`nifty.graph.nodeWeightedWatershedsSegmentation`.

    
    Args:
        heightMap (numpy.ndarray) : height / evaluation map  
        seeds (numpy.ndarray) : Seeds as non zero elements in the array.
            (default: {nifty.segmentation.localMinimaSeeds(heightMap)})
        method (str): Algorithm type can be:

            *   "node_weighted": ordinary node weighted watershed
            *   "edge_weighted": edge weighted watershed (minimum spanning tree)
        
            (default: {"max"})

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
    if seeds is None:
        seeds = localMinimaSeeds(heightMap)

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

def distanceTransformWatersheds(pmap, preBinarizationMedianRadius=1, threshold = 0.5, preSeedSigma=0.75):
    """Superpixels for neuro data as in http://brainiac2.mit.edu/isbi_challenge/
    
    Use raw data and membrane probability maps to
    generate a over-segmentation suitable for neuro data
    
    Args:
        pmap (numpy.ndarray): Membrane probability in [0,1].
        preBinarizationMedianRadius (int) : Radius of
            diskMedian filter applied to the probability map
            before binarization. (default:{1})
        threshold (float) : threshold to binarize 
            probability map  before applying
            the distance transform (default: {0.5})
        preSeedSigma (float) : smooth the distance
            transform image before getting the seeds.
       
    Raises:
        RuntimeError: if applied to data with wrong dimensionality
    """


    if pmap.ndim != 2:
        raise RuntimeError("Currently only implemented for 2D data")




    # pre-process pmap  / smooth pmap
    if preBinarizationMedianRadius >= 1 :
        toBinarize = filters.diskMedian(pmap, radius=preBinarizationMedianRadius)
        toBinarize -= toBinarize.min()
        toBinarize /= toBinarize.max()
    else:
        toBinarize = pmap


    # computing the distance transform inside and outside
    b1 = toBinarize < threshold

    #b0 = b1 == False

    dt1 = scipy.ndimage.morphology.distance_transform_edt(b1)
    #dt0 = scipy.ndimage.morphology.distance_transform_edt(b0)

    if preSeedSigma > 0.01:
        toSeedOn = filters.gaussianSmoothing(dt1, preSeedSigma)
    else:
        toSeedOn = dt1
    # find the seeds
    seeds  = localMaximaSeeds(toSeedOn)

    # compute  growing map
    a = filters.gaussianSmoothing(pmap, 0.75)
    b = filters.gaussianSmoothing(pmap, 3.00)
    c = filters.gaussianSmoothing(pmap, 9.00)
    d =  growMapB = numpy.exp(-0.2*dt1)
    growMap = (7.0*a + 4.0*b + 2.0*c + 1.0*d)/14.0


    # grow regions
    seg = seededWatersheds(growMap, seeds=seeds,
        method='edge_weighted',  acc='interpixel')
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

    return  __peak_local_max(image, exclude_border=False, indices=False)

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
    return localMaximaSeeds(-1.0 * image)
    
def localMaximaSeeds(image):
    """Get seed from local maxima
    
    Get seeds by running connected components 
    on the local maxima.
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
        raise RuntimeError("localMaximaSeeds is currently only implemented for 2D images")

    lm = localMaxima(image)
    cc = connectedComponents(lm, dense=True, ignoreBackground=True)
    return cc
    
def markBoundaries(image, segmentation, color=None, thin=True):
    """Mark the boundaries in an image
    
    Mark boundaries in an image.

    Warning:

        The returned image shape is twice as large
        as the input if this is True.
    
    Args:
        image:  the input image 
        segmentation:  the segmentation
        color (tuple) : the edge color(default: {(0,0,0)})
        thin (bool) : IF true, the image is interpolated and 
            the boundaries are marked in the interpolated
            image. This will make the output twice as large.
    Returns:
        (numpy.ndarray) : image with marked boundaries. Note that
            result image has twice as large shape as the input if thin is True.
    """
    if color is None:
        color = (0,0,0)
    if thin:
        shape = segmentation.shape
        img2 =   __imresize(image, [2*s for s in shape])#, interp='nearest')
        #img2 = __zoom(segmentation.astype('float32'), 2, order=1)
        seg2 = __zoom(segmentation.astype('float32'), 2, order=0)
        seg2 = seg2.astype('uint32')

        return skimage.segmentation.mark_boundaries(img2, seg2.astype('uint32'), color=color)
    else:
        return skimage.segmentation.mark_boundaries(image, segmentation.astype('uint32'), color=color)

def segmentOverlay(image, segmentation, beta=0.5, zeroToZero=False, showBoundaries=True, color=None, thin=True):

    cmap = numpy.random.rand (int(segmentation.max()+1), 3)

    if zeroToZero:
        cmap[0,:] = 0
    cSeg = numpy.take(cmap, numpy.require(segmentation,dtype='int64'),axis=0)

    imgCp = image.astype('float32')
    if imgCp.ndim != 3:
        imgCp  = numpy.concatenate([imgCp[:,:,None]]*3,axis=2)


    mi = imgCp.min()
    ma = imgCp.max()

    if(ma-mi > 0.000001):
        imgCp -= mi 
        imgCp /= (ma - mi)

    overlayImg =  (1.0-beta)*imgCp + (beta)*cSeg

    if showBoundaries:
        return markBoundaries(overlayImg, segmentation, color=color, thin=thin)
    else:
        return overlayImg





def randomColormap(size=10000, zeroToZero=False):
    cmap = numpy.random.rand (int(size),3)
    if zeroToZero:
        cmap[0,:] = 0
    cmap = __ListedColormap(cmap)
    return cmap
