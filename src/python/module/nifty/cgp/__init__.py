from __future__ import absolute_import
from __future__ import print_function
from .import _cgp as __cgp
from ._cgp import *

from scipy.ndimage import grey_dilation
from skimage.transform import resize as __imresize
from scipy.ndimage import zoom as __zoom

try:
    import pylab
    import matplotlib.cm as cm
    __hasPyLabAndMatplotlib = True
except ImportError:
    __hasPyLabAndMatplotlib = False

__all__ = [
    'makeCellImage'
]

for key in __cgp.__dict__.keys():
    __all__.append(key)
    try:
         __cgp.__dict__[key].__module__='nifty.cgp'
    except:
        pass

import numpy

def _extenTGrid():


    def _gridView(self):
        a = self._gridView()
        a.flags.writeable = False
        return a

    TopologicalGrid2D.__array__ = _gridView

    def extractCellsBounds(self):
        return Bounds2D(self)

    def extractCellsGeometry(self, fill=True, sort1Cells=True):
        return Geometry2D(self, fill=fill,sort1Cells=sort1Cells)

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


    # cell 1 bounded by (in 2D junctions of boundaries)
    def get_inverse1(self):
        return Cell1BoundedByVector2D(self)
    Cell0BoundsVector2D.reverseMapping = get_inverse1

    # cell 1 bounded by (in 2D junctions of boundaries)
    def get_inverse2(self):
        return Cell2BoundedByVector2D(self)
    Cell1BoundsVector2D.reverseMapping = get_inverse2


__extend__()
del __extend__



def makeCellImage(image, mask_image, lut, size=None):
    if image.shape[0:2] != mask_image.shape:
        image =   __imresize(image, mask_image.shape)#, interp='nearest')

    if size is not None:
        mask_image = grey_dilation(mask_image, size=size)
    if(not __hasPyLabAndMatplotlib):
        raise RuntimeError("showCellValues")
    else:
        if lut.ndim ==1:
            nLutChannels = 1
            zeroValue = 0
            _lut = numpy.hstack((zeroValue,lut))
            lutImg = numpy.take(_lut, mask_image)
            resImage = image.copy()
            whereImage = mask_image!=0
            resImage[whereImage] = lutImg[whereImage]
            return resImage


        elif lut.ndim ==2:
            lutC = lut.shape[1]
            if image.ndim == 2:
                image = numpy.concatenate([image[:,:,None]]*lutC,axis=2)

            nLutChannels = lut.shape[1]
            zeroValue = [0]*nLutChannels
            zeroValue = numpy.array(zeroValue)[None,:]
            _lut = numpy.concatenate((zeroValue,lut))

            #lutImg0
            #lutImg1
            #lutImg2




            lutImg = _lut[mask_image.ravel(),:].reshape(mask_image.shape+(lutC,))


            resImage = image.copy()
            whereImage = mask_image!=0
            resImage[whereImage] = lutImg[whereImage]
            return resImage


            #resImage = image.copy()
            #print("resImg",resImage.shape)
            #whereImage = mask_image!=0
            #for c in range(lutC):
            #    resImgC = resImage[:,:,c]
            #    lutImgC = lutImg[:,:,c]
            #    resImgC[whereImage] = lutImgC[whereImage]
            #return resImage

        else:
            raise ValueError("lut ndim must be in [1,2]")

        #print("theklut",_lut.shape,mask_image.shape)
        lutImg = numpy.take(_lut, mask_image)
        lutImg = _lut[mask_image.ravel(),:].reshape(mask_image.shape+(3,))
        #print("lut image",lutImg.shape)
        resImage = image.copy()
        whereImage = mask_image!=0
        resImage[whereImage] = lutImg[whereImage]
        return resImage






def cell1Features(tgrid, geometry=None, bounds=None,
        boundedBy=None,raw=None, pmap=None):


    if geometry is None:
        geometry = tgrid.extractCellsGeometry()
    if geometry is None:
        bounds = tgrid.extractCellsBounds()
    if boundedBy is None:
        boundedBy = {1:bounds[0].reverseMapping(),
                     2:bounds[1].reverseMapping()}


    feats = []
    names = []

    # curvature
    op = Cell1CurvatureFeatures2D()
    feat = op(cell1GeometryVector=geometry[1],
              cell1BoundedByVector=boundedBy[1])
    feats.append(feat)
    fNames = op.names()

    assert len(fNames) == feat.shape[1]

    names.extend(fNames)

    # line segment dist
    op = Cell1LineSegmentDist2D()
    feat = op(cell1GeometryVector=geometry[1])
    feats.append(feat)
    fNames = op.names()
    assert len(fNames) == feat.shape[1]
    names.extend(fNames)

    # basic geometric features
    op = Cell1BasicGeometricFeatures2D()
    feat = op(cell1GeometryVector=geometry[1],
             cell2GeometryVector=geometry[2],
             cell1BoundsVector=bounds[1])
    feats.append(feat)
    fNames = op.names()
    print(feat.shape, len(fNames))
    assert len(fNames) == feat.shape[1]
    names.extend(fNames)

    # basic topological features
    op = Cell1BasicTopologicalFeatures2D()
    feat = op(cell0BoundsVector=bounds[0],
              cell1BoundsVector=bounds[1],
              cell1BoundedByVector=boundedBy[1],
              cell2BoundedByVector=boundedBy[2])
    feats.append(feat)
    fNames = op.names()
    assert len(fNames) == feat.shape[1]
    names.extend(fNames)

    feats =  numpy.concatenate(feats, axis=1)
    assert len(names) == feats.shape[1]

    return feats, names


