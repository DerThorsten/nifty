from __future__ import absolute_import
from ._filters import *


__all__ = []
for key in _filters.__dict__.keys():
    __all__.append(key)




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