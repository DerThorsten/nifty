from  __future__ import print_function,division

import unittest
import nifty
import nifty.cgp as ncgp

import numpy

seg = [
    [1,1,2],
    [1,3,1],
    [1,1,1],
    [4,4,5],
    [6,6,7],
]
seg = numpy.array(seg,dtype='uint32')

tgrid = ncgp.TopologicalGrid2D(seg)
ftgrid = ncgp.FilledTopologicalGrid2D(tgrid)
numberOfCells = tgrid.numberOfCells




tGridArray = numpy.array(tgrid) 
print(tGridArray)

tGridArray = numpy.array(ftgrid) 
print(tGridArray)

cellBounds = ncgp.Bounds2D(tgrid)

cell0Bounds = numpy.array(cellBounds[0])
cell1Bounds = numpy.array(cellBounds[1])


cellGeometry = ncgp.Geometry2D(tgrid)

for cellType in (0,1,2):
    
    nCells = numberOfCells[cellType]

    print("cell-%d (#%d):"%(cellType, nCells))
    cellsGeo = cellGeometry[cellType] 
    for cellIndex in range(nCells):
        cellGeo =  numpy.array(cellsGeo[cellIndex])
        print(cellIndex,":\n",cellGeo)



if False:
    for celLType in (0,1):
        nCells = tgrid.numberOfCells(celLType)
        print("%d-Cell:"%celLType)
        cellBounds = bounds[celLType]
        for c in range(nCells):
            print("  ",[i for i in cellBounds[c]])
            



