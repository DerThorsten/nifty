from __future__ import absolute_import
from ._filters import *

import numpy
import scipy.ndimage.filters

import skimage.filters
import skimage.morphology

__all__ = []
for key in _filters.__dict__.keys():
    __all__.append(key)



def gaussianSmoothing(image, sigma, nSpatialDimensions=2):
    image = numpy.require(image, dtype='float32')
    if image.ndim == nSpatialDimensions:
        return scipy.ndimage.filters.gaussian_filter(image, sigma)
    elif image.ndim ==  nSpatialDimensions + 1:
        raise RuntimeError("not yer implemented")
    else:
        raise RuntimeError("image dimension does not match spatial dimension")

def gaussianGradientMagnitude(image, sigma, nSpatialDimensions=2):
    image = numpy.require(image, dtype='float32')
    if image.ndim == nSpatialDimensions:
        return scipy.ndimage.filters.gaussian_gradient_magnitude(image, sigma)
    elif image.ndim ==  nSpatialDimensions + 1:
        out = None
        nChannels = image.shape[image.ndim-1]
        for c in range(nChannels):
            cImage = image[...,c]
            gm = scipy.ndimage.filters.gaussian_gradient_magnitude(image, sigma)
            if out is None:
                out = gm
            else:
                out += gm
            out /= image.nChannels
        return out
    else:
        raise RuntimeError("image dimension does not match spatial dimension")
    

def affinitiesToProbability(affinities, edge_format=-1):
    ndim = affinities.ndim
    n_channels = affinities.shape[2]
    if ndim != 3 or n_channels != 2:
        raise RuntimeError("ndim must be 3 and n_channels must be 2")


    if edge_format == 1:
        ax = affinities[:, :, 0]
        ay = affinities[:, :, 1]

        ax_ = ax[0:-1,:   ]
        ay_ = ax[:   ,0:-1]


        axx = ax.copy()
        ayy = ay.copy()

        axx[1 :, :] += ax_
        ayy[:, 1 :] += ay_

    elif edge_format == -1:
        ax = affinities[:, :, 0]
        ay = affinities[:, :, 1]

        ax_ = ax[1:,:   ]
        ay_ = ax[:   ,1:]


        axx = ax.copy()
        ayy = ay.copy()

        axx[0:-1, :] += ax_
        ayy[:, 0:-1] += ay_
    
    else:
        raise RuntimeError("format must be in [1,-1]")


    return 1- (axx + ayy)/2.0



try :
    import vigra
    __has_vigra = True
except ImportError:
    __has_vigra = False


def diskMedian(img, radius):
    nimg = img.copy()
    oldMin = img.min()
    oldMax = img.max()
    nimg = numpy.require(img, dtype='float32')
    nimg -= oldMin
    nimg /= (oldMax - oldMin)
    nimg *= 255.0
    nimg = nimg.astype('uint8')
    disk  = skimage.morphology.disk(radius)
    r = skimage.filters.median(nimg, disk).astype('float32')/255.0
    r *= (oldMax - oldMin)
    r += oldMin
    return r




if __has_vigra:

    def hessianOfGaussianEigenvalues(image, sigma):

        imageShape = image.shape
        nDim = image.ndim 
        
        iamgeR = numpy.require(image, dtype='float32', requirements=['C'])
        imageT = iamgeR.T

        res = vigra.filters.hessianOfGaussianEigenvalues(imageT, sigma).view(numpy.ndarray).T
        res = numpy.moveaxis(res,0,-1)
        
        return numpy.require(res, requirements=['C'])



    def hessianOfGaussianStrongestEigenvalue(image, sigma):

        imageShape = image.shape
        nDim = image.ndim 
        
        iamgeR = numpy.require(image, dtype='float32', requirements=['C'])
        imageT = iamgeR.T

        res = vigra.filters.hessianOfGaussianEigenvalues(imageT, sigma)[:,:,0].view(numpy.ndarray).T

        return numpy.require(res, requirements=['C'])