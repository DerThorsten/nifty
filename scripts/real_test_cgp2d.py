from  __future__ import print_function,division

import nifty
import nifty.cgp as ncgp
from nifty.ground_truth import Overlap
import h5py
import numpy
import fastfilters as ffilt
import vigra
from functools import partial

import matplotlib,numpy
import pylab

# A random colormap for matplotlib
#cmap = matplotlib.colors.ListedColormap ( numpy.random.rand ( 10000,3))


def watersheds(raw, sigma):
    edgeIndicator = ffilt.hessianOfGaussianEigenvalues(raw, sigma)[:,:,0]
    seg, nseg = vigra.analysis.watersheds(edgeIndicator)
    #seg -= 1
    #vigra.segShow(raw, seg)
    #vigra.show()
    return seg,nseg





class Extractor(object):

    def __init__(self, raw, seg, gt=None, patchSize=[100,100],**kwargs):
        
        #raw input data
        self.raw = numpy.squeeze(raw)
        self.seg = numpy.squeeze(seg)
        self.gt  = numpy.squeeze(gt)

        # shorthands
        self.shape = self.raw.shape

        # settings
        self.patchSize = patchSize

        # apply paddings
        ps = self.patchSize
        padding = ((ps[0], ps[0]),(ps[1],ps[1]))
        pad = partial(numpy.pad,pad_width=padding)

        self.paddingMask = pad(numpy.zeros(self.shape), mode='constant',constant_values=1)
        self.paddedRaw = pad(self.raw, mode='reflect')
        self.paddedSeg = vigra.analysis.labelImage(pad(self.seg, mode='reflect')).squeeze()
        self.paddedGt  = None
        if self.gt is not None:
            self.paddedGt = vigra.analysis.labelImage(pad(self.gt, mode='reflect')).squeeze()
            #self.paddedGt = pad(self.gt + 1, mode='constant', constant_values=0)



        # compute cgp
        self.tGrid = ncgp.TopologicalGrid2D(self.paddedSeg)
        self.tShape = self.tGrid.topologicalGridShape
        self.numberOfCells = self.tGrid.numberOfCells
        self.fGrid = ncgp.FilledTopologicalGrid2D(self.tGrid)
        self.cellGrids = [
            self.fGrid.cellMask([1,0,0]),
            self.fGrid.cellMask([0,1,0]),
            None
        ]
        self.cellGrids[0][self.cellGrids[0]!=0] -= self.fGrid.cellTypeOffset[0]
        self.cellGrids[1][self.cellGrids[1]!=0] -= self.fGrid.cellTypeOffset[1]

        self.cellMasks = [numpy.clip(x,0,1) for x in self.cellGrids[0:2]]+[None]
        rawT = vigra.sampling.resize(self.paddedRaw, self.tShape)
        #rawT[self.cellMasks[1]!=0] = 0



                
        self.cellsBounds   = self.tGrid.extractCellsBounds()
        self.cellsGeometry = self.tGrid.extractCellsGeometry(fill=True)
        self.cells1Bounds = self.cellsBounds[1]
        self.cells1Geometry = self.cellsGeometry[1]


        # center of mass
        self.cell1CentersOfMass = self.cells1Geometry.centersOfMass();

        print(self.cell1CentersOfMass)
        
        # compute gt    
        ol = Overlap(self.numberOfCells[2], self.paddedSeg, self.paddedGt)
        cell1Probs = ol.differentOverlaps(numpy.array(self.cells1Bounds))


        with  nifty.Timer("makeCellImage"):
            probImg = ncgp.makeCellImage(rawT, self.cellGrids[1], cell1Probs*255.0)



        vigra.imshow(probImg.astype('uint8'),cmap='gist_heat')
        vigra.show()

        if False:
            vigra.imshow(self.paddingMask)
            vigra.show()

            vigra.imshow(self.paddedRaw)
            vigra.show()

            vigra.segShow(self.paddedRaw, self.paddedSeg)
            vigra.show()

            vigra.segShow(self.paddedRaw, self.paddedGt)
            vigra.show()



# cremi data
f = "/home/tbeier/Downloads/sample_A_20160501.hdf"
f = h5py.File(f)
raw = f['volumes/raw'][0,:,:].astype('float32')[0:500,0:500].squeeze()
gt  = f['volumes/labels/neuron_ids'][0,:,:].astype('uint32')[0:500,0:500]
seg, nseg = watersheds(raw, 3.0)

patchSize = [50, 50]
extractor = Extractor(raw=raw, seg=seg, gt=gt, patchSize=patchSize)


sys.exit(0)







seg = numpy.array(seg,dtype='uint32')
#vigra.segShow(raw,seg)
#vigra.show()


tGrid = ncgp.TopologicalGrid2D(seg)
numberOfCells = tGrid.numberOfCells

fGrid = ncgp.FilledTopologicalGrid2D(tGrid)
cell1Mask = numpy.clip(fGrid.cellMask([1,1,0]),0,1)
vigra.imshow(cell1Mask)
vigra.show()





cells0Bounds = numpy.array(cellBounds[0])
cells1Bounds = numpy.array(cellBounds[1])

cells0Geo = cellGeometry[0]
cells1Geo = cellGeometry[1]
cells2Geo = cellGeometry[2]


# loop over all 1-cells (boundaries)

for cell1Index  in range(numberOfCells[1]):

    #print(cells1Bounds[cell1Index,:])

    cell1Geo = cells1Geo[cell1Index]
    print(numpy.array(cell1Geo).shape)