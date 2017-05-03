from __future__ import absolute_import,print_function
from ._ground_truth import *

import numpy
from scipy import ndimage as ndi

__all__ = []
for key in _ground_truth.__dict__.keys():
    __all__.append(key)





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
