from __future__ import print_function
from _cgp import *


try:
    import pylab
    import matplotlib.cm as cm
    __hasPyLabAandMatplotlib = True
except ImportError:
    __hasPyLabAandMatplotlib = False

__all__ = []

for key in _cgp.__dict__.keys():
    __all__.append(key)


import numpy 

def _extenTGrid():
   

    def _gridView(self):
        a = self._gridView()
        a.flags.writeable = False
        return a

    TopologicalGrid2D.__array__ = _gridView
    
    def extractCellsBounds(self):
        return Bounds2D(self)

    def extractCellsGeometry(self, fill=False):
        return Geometry2D(self, fill=fill)

    TopologicalGrid2D.extractCellsBounds = extractCellsBounds
    TopologicalGrid2D.extractCellsGeometry = extractCellsGeometry


    # filled tgrid

    FilledTopologicalGrid2D.__array__ = _gridView

    def cellMask2D(self, showCells):
        offset = self.cellTypeOffset
        a = numpy.array(self, copy=True)

        for cellType in [0,1,2]:
            if not showCells[cellType]:
                if cellType==0:
                    a[a>offset[0]]  = 0
                else:
                   a[ numpy.logical_and(a>offset[cellType], a<=offset[cellType-1])] = 0 


        return a

    FilledTopologicalGrid2D.cellMask = cellMask2D

_extenTGrid()
del _extenTGrid



def __extend__():
    def getGeoItem(self, cellType):
        if cellType == 0:
            return self.cell0Geometry()
        elif cellType == 1:
            return  self.cell1Geometry()
        elif cellType == 2:
            return  self.cell2Geometry()
        else:
            return IndexError("cellType must be 0 1 or 2")

    Geometry2D.__getitem__ = getGeoItem


    def getBoundsItem(self, cellType):
        if cellType == 0:
            return self.cell0Bounds()
        elif cellType == 1:
            return  self.cell1Bounds()
        else:
            return IndexError("cellType must be 0 or 1")

    Bounds2D.__getitem__ = getBoundsItem

__extend__()
del __extend__





def makeCellImage(image, mask_image, lut):
    if(not __hasPyLabAandMatplotlib):
        raise RuntimeError("showCellValues")
    else:
        if lut.ndim ==1:
            nLutChannels = 1
            zeroValue = 0
        elif lut.ndim ==2:
            nLutChannels
            zeroValue = [0]*nLutChannels
        else:
            raise ValueError("lut ndim must be in [0,1]")
        _lut = numpy.hstack((zeroValue,lut))
        lutImg = numpy.take(_lut, mask_image)
        print("lut image",lutImg.shape)
        resImage = image.copy()
        whereImage = mask_image!=0
        resImage[whereImage] = lutImg[whereImage]
        return resImage

# class Cgp(object):
#     def __init__(self, labels):
#         self.labels = labels.squeeze()
#         if self.labels.ndim == 2:
#             pass
#         else:
#             raise RuntimeError("not yet implemented")
