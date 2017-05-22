"""
Cgp Cartesian Grid Partitioning 2D
====================================

Cgp Cartesian Grid Partitioning 2D
"""
from __future__ import print_function

# numpy
import numpy

# skimage
import skimage.filters       # filters
import skimage.segmentation  # Superpixels
import skimage.data          # Data
import skimage.color         # rgb2Gray

# pylab
import pylab                # Plotting
import matplotlib

# increase default figure size
a,b = pylab.rcParams['figure.figsize']
pylab.rcParams['figure.figsize'] = 2.0*a, 2.0*b
    

import sys; sys.path.append("/home/tbeier/bld/nifty/python/")
# nifty
import nifty
import nifty.cgp     

cellNames = ['Junctions','Edges','Regions']




############################################################################
# Load image and compute over-segmentation
img = skimage.data.coins()
pylab.imshow(img)

############################################################################
# Superpixels
overseg = skimage.segmentation.slic(img, n_segments=2000,
    compactness=0.1, sigma=1)
# let overseg start from 1
overseg += 1 
assert overseg.min() == 1
cmap = numpy.random.rand ( overseg.max()+1,3)
cmap = matplotlib.colors.ListedColormap(cmap)
pylab.imshow(overseg, cmap=cmap)
pylab.show()

############################################################################
# Compute cgp / topological grid 
tgrid = nifty.cgp.TopologicalGrid2D(overseg)



############################################################################
# Show cells
ftgrid = nifty.cgp.FilledTopologicalGrid2D(tgrid)
f = pylab.figure()

f.add_subplot(2, 2, 1)
pylab.imshow(img,cmap='gray')
pylab.title('Raw Data')
cmap = numpy.random.rand ( 100000,3)
cmap[0,:] = 0
cmap = matplotlib.colors.ListedColormap(cmap)
for i,cellType in enumerate((2,1,0)):
    showCells = [False,False,False]
    showCells[cellType] = True
    cellMask = ftgrid.cellMask(showCells)
    cellMask[cellMask!=0] -=  ftgrid.cellTypeOffset[cellType]

    f.add_subplot(2, 2, i+2)
    pylab.imshow(cellMask,cmap=cmap)
    pylab.title('Cell-%d Labels / \n%s Labels '%(cellType,cellNames[cellType] )  )
pylab.show()


############################################################################
# Compute edge strength

smoothed = skimage.filters.gaussian(img, 2.5)
edgeStrength = skimage.filters.sobel(smoothed)

pylab.imshow(edgeStrength)
pylab.show()