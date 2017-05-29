from __future__ import absolute_import
from . import _rag as __rag
from ._rag import *
from .. import Configuration

import numpy


__all__ = []
for key in __rag.__dict__.keys():
    try:
        __rag.__dict__[key].__module__='nifty.graph.rag'
    except:
        pass

    __all__.append(key)


def gridRag(labels, numberOfThreads=-1, serialization = None):
    labels = numpy.require(labels ,dtype='uint32')


    if numpy.squeeze(labels).ndim == 2:
        if serialization is None:
            return explicitLabelsGridRag2D(labels, numberOfThreads=int(numberOfThreads))
        else:
            return explicitLabelsGridRag2D(labels, serialization)

    elif numpy.squeeze(labels).ndim == 3:
        if serialization is None:
            return explicitLabelsGridRag3D(labels, numberOfThreads=int(numberOfThreads))
        else:
            return explicitLabelsGridRag3D(labels, serialization)
    else:
        raise RuntimeError("wrong dimension, currently only 2D and 3D is implemented")


def gridRagStacked2D(labels, numberOfThreads = -1):
    labels = numpy.require(labels)

    return gridRagStacked2DExplicitImpl(labels, numberOfThreads)


# returns the edge coordinates as
# topological coordinates (returnTopologicalCoordinates == True)
# or pixel coordinates (returnTopologicalCoordinates == False)
def edgeCoordinates(rag,
        returnTopologicalCoordinates = False,
        numberOfThreads = -1):

    # this returns the coordinates as flat lists and
    # topological (-> sum of pixel coordinates)
    coordinates = edgeCoordinatesImpl(rag, numberOfThreads)

    # go from flat list to correct numpy arrays
    dim = len(rag.shape)
    coordinates = [
            numpy.array(coord, dtype = 'int64').reshape( (len(coord)/dim, dim) ) for coord in coordinates ]

    if returnTopologicalCoordinates:
        return coordinates
    else:
        coordinates = [ numpy.divide(coord.astype('float64'), 2)  for coord in coordinates ]
        coordinates = [ numpy.concatenate(
            [func(coord).astype('int64') for func in (numpy.floor, numpy.ceil)],
            axis = 0) for coord in coordinates ]
        return coordinates

if Configuration.WITH_HDF5:

    def gridRagHdf5(labels, numberOfLabels, blockShape = None, numberOfThreads=-1):

        dim = labels.ndim
        if blockShape is None:
            bs = [100]*dim
        else:
            bs = blockShape

        if dim == 2:
            labelsProxy = gridRag2DHdf5LabelsProxy(labels, int(numberOfLabels))
            ragGraph = gridRag2DHdf5(labelsProxy,bs,int(numberOfThreads))
        elif dim == 3:
            labelsProxy = gridRag3DHdf5LabelsProxy(labels, int(numberOfLabels))
            ragGraph = gridRag3DHdf5(labelsProxy,bs,int(numberOfThreads))
        else:
            raise RuntimeError("gridRagHdf5 is only implemented for 2D and 3D not for %dD"%dim)

        return ragGraph

    def gridRagStacked2DHdf5(labels, numberOfLabels, numberOfThreads=-1, serialization = None):
        dim = labels.ndim
        if dim == 3:
            labelsProxy = gridRag3DHdf5LabelsProxy(labels, int(numberOfLabels))
            if serialization is not None:
                ragGraph = gridRagStacked2DHdf5Impl(labelsProxy,serialization)
            else:
                ragGraph = gridRagStacked2DHdf5Impl(labelsProxy,int(numberOfThreads))
        else:
            raise RuntimeError("gridRagStacked2DHdf5 is only implemented for 3D not for %dD"%dim)

        return ragGraph

    def writeStackedRagToHdf5(rag, savePath):
        import vigra
        vigra.writeHDF5(rag.numberOfNodes, savePath, 'numberOfNodes')
        vigra.writeHDF5(rag.numberOfEdges, savePath, 'numberOfEdges')
        vigra.writeHDF5(rag.uvIds(),       savePath, 'uvIds')
        vigra.writeHDF5(rag.minMaxLabelPerSlice(), savePath, 'minMaxLabelPerSlice')
        vigra.writeHDF5(rag.numberOfNodesPerSlice(), savePath, 'numberOfNodesPerSlice')
        vigra.writeHDF5(rag.numberOfInSliceEdges(), savePath, 'numberOfInSliceEdges')
        vigra.writeHDF5(rag.numberOfInBetweenSliceEdges(), savePath, 'numberOfInBetweenSliceEdges')
        vigra.writeHDF5(rag.inSliceEdgeOffset(), savePath, 'inSliceEdgeOffset')
        vigra.writeHDF5(rag.betweenSliceEdgeOffset(), savePath, 'betweenSliceEdgeOffset')
        vigra.writeHDF5(rag.totalNumberOfInSliceEdges, savePath, 'totalNumberOfInSliceEdges')
        vigra.writeHDF5(rag.totalNumberOfInBetweenSliceEdges, savePath, 'totalNumberOfInBetweenSliceEdges')
        vigra.writeHDF5(rag.edgeLengths(), savePath, 'edgeLengths')


    def readStackedRagFromHdf5(labels, numberOfLabels, savePath):
        assert labels.ndim == 3
        import vigra
        # load the serialization from h5

        # serialization of the undirected graph
        serialization = numpy.array(vigra.readHDF5(savePath, 'numberOfNodes'))
        serialization = numpy.append(serialization, numpy.array(vigra.readHDF5(savePath, 'numberOfEdges')) )
        serialization = numpy.append(serialization, vigra.readHDF5(savePath, 'uvIds').ravel() )

        # serialization of the stacked rag
        serialization = numpy.append(serialization, numpy.array(vigra.readHDF5(savePath, 'totalNumberOfInSliceEdges')) )
        serialization = numpy.append(serialization, numpy.array(vigra.readHDF5(savePath, 'totalNumberOfInBetweenSliceEdges')) )
        # load all the per slice data to squeeze it in the format we need for serializing
        # cf. nifty/include/nifty/graph/rag/grid_rag_stacked_2d.hxx serialize
        inSliceDataKeys = ['numberOfInSliceEdges', 'numberOfInBetweenSliceEdges', 'inSliceEdgeOffset', 'betweenSliceEdgeOffset']
        perSliceData = numpy.concatenate( [vigra.readHDF5(savePath, key)[:,None] for key in inSliceDataKeys], axis = 1 )
        perSliceData = numpy.concatenate( [perSliceData, vigra.readHDF5(savePath, 'minMaxLabelPerSlice') ], axis = 1)
        serialization = numpy.append(serialization, perSliceData.ravel() )
        serialization = numpy.append(serialization, vigra.readHDF5(savePath, 'edgeLengths') )

        # get the rag from serialization + labels
        labelsProxy = gridRag3DHdf5LabelsProxy(labels, int(numberOfLabels))
        return gridRagStacked2DHdf5Impl(labelsProxy, serialization.astype('uint64'))

