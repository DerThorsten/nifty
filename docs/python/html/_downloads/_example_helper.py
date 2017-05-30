"""
Helpers for examples
====================================

Helper functions shared by examples
"""

import pylab

# increase default figure size

a,b = pylab.rcParams['figure.figsize']
pylab.rcParams['figure.figsize'] = 1.5*a, 1.5*b


def randColorMap(size, zeroToZeros=True):
    r = numpy.random.rand ( size,3)
    if zeroToZeros:
        r[0,:] = 0
    return matplotlib.colors.ListedColormap ( r)