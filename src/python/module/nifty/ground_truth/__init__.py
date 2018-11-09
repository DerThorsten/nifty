from __future__ import absolute_import,print_function
from . import _ground_truth as __ground_truth
from ._ground_truth import *

import numpy
from scipy import ndimage as ndi

__all__ = []
for key in __ground_truth.__dict__.keys():
    __all__.append(key)

    try:
        __ground_truth.__dict__[key].__module__='nifty.ground_truth'
    except:
        pass




def thinSegFilter(seg, sigma, radius=None):
    edges = segToEdges2D(seg)
    dt = ndi.distance_transform_edt(1 - edges)
    if radius is None:
        radius = int(3.5*float(sigma) + 0.5)
        radius += 4


    p_seg = numpy.pad(seg,radius,mode='reflect').astype('uint32')
    p_dt  = numpy.pad(dt,radius ,mode='reflect').astype('uint32')

    print("p_seg",p_seg.shape,p_seg.dtype)
    print("p_dt",p_dt.shape,p_dt.dtype)
    print("sigma",sigma)
    print("radius",radius)

    out = _ground_truth._thinSegFilter(p_seg, p_dt, sigma=float(sigma), radius=radius)
    return out[radius:radius+seg.shape[0],radius:radius+seg.shape[1]],dt




def overlap(segmentation, groundTruth):
    """factory function for :class:`nifty.ground_truth.Overlap`

    create an instance of :class:`nifty.ground_truth.Overlap`
    which can be used to project ground truth
    to some segmentation / over-segmentation

    Args:
        segmentation (numpy.ndarray): The segmentation / over-segmentation
        groundTruth (numpy.ndarray): The ground truth as node labeling.


    Returns:
        [description]
        [type]
    """
    a = numpy.require(segmentation, dtype='uint64')
    b = numpy.require(groundTruth, dtype='uint64')
    return Overlap(a.max(), a, b)
