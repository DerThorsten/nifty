from __future__ import absolute_import

import os
from .. import Configuration

import numpy

from . import _rag as __rag
from ._rag import *

__all__ = []
for key in __rag.__dict__.keys():
    try:
        __rag.__dict__[key].__module__ = 'nifty.graph.rag'
    except:
        pass

    __all__.append(key)

try:
    import h5py
    WITH_H5PY = True
except ImportError:
    WITH_H5PY = False

if Configuration.WITH_Z5:
    import nifty.z5


def gridRag(labels, numberOfLabels, blockShape=None, numberOfThreads=-1, serialization=None):
    labels = numpy.require(labels, dtype='uint32')
    dim = labels.ndim
    blockShape_ = [100] * dim if blockShape is None else blockShape

    if dim == 2:
        if serialization is None:
            return explicitLabelsGridRag2D(labels,
                                           blockShape=blockShape_,
                                           numberOfLabels=numberOfLabels,
                                           numberOfThreads=int(numberOfThreads))
        else:
            return explicitLabelsGridRag2D(labels,
                                           numberOfLabels=numberOfLabels,
                                           serialization=serialization)

    elif dim == 3:
        if serialization is None:
            return explicitLabelsGridRag3D(labels,
                                           blockShape=blockShape_,
                                           numberOfLabels=numberOfLabels,
                                           numberOfThreads=int(numberOfThreads))
        else:
            return explicitLabelsGridRag3D(labelsProxy,
                                           numberOfLabels=numberOfLabels,
                                           serialization=serialization)

    else:
        raise RuntimeError("wrong dimension, currently only 2D and 3D is implemented")


def gridRagStacked2D(labels, numberOfLabels, serialization=None, numberOfThreads=-1):
    labels = numpy.require(labels, dtype='uint32')
    assert labels.ndim == 3, "Stacked rag is only available for 3D labels"
    if serialization is None:
        return gridRagStacked2DExplicitImpl(labels,
                                            numberOfLabels=numberOfLabels,
                                            numberOfThreads=numberOfThreads)
    else:
        return gridRagStacked2DExplicitImpl(labels,
                                            numberOfLabels=numberOfLabels,
                                            serialization=serialization)


# helper class for rag coordinates
def ragCoordinates(rag, numberOfThreads=-1):
    if len(rag.shape) == 2:
        return coordinatesFactoryExplicit2d(rag, numberOfThreads=numberOfThreads)
    else:
        return coordinatesFactoryExplicit3d(rag, numberOfThreads=numberOfThreads)


def ragCoordinatesStacked(rag, numberOfThreads=-1):
    return coordinatesFactoryStackedRag3d(rag, numberOfThreads=numberOfThreads)


if Configuration.WITH_HDF5:

    def gridRagHdf5(labels, numberOfLabels, blockShape=None, numberOfThreads=-1):

        dim = labels.ndim
        blockShape_ = [100] * dim if blockShape is None else blockShape

        if dim == 2:
            return gridRag2DHdf5(labels,
                                 numberOfLabels=numberOfLabels,
                                 blockShape=blockShape_,
                                 numberOfThreads=int(numberOfThreads))
        elif dim == 3:
            return gridRag3DHdf5(labels,
                                 numberOfLabels=numberOfLabels,
                                 blockShape=blockShape_,
                                 numberOfThreads=int(numberOfThreads))
        else:
            raise RuntimeError("gridRagHdf5 is only implemented for 2D and 3D not for %dD" % dim)


    def gridRagStacked2DHdf5(labels, numberOfLabels, numberOfThreads=-1, serialization=None):
        assert labels.ndim == 3, "Stacked rag is only available for 3D labels"
        if serialization is None:
            return gridRagStacked2DHdf5Impl(labels,
                                            numberOfLabels=numberOfLabels,
                                            numberOfThreads=int(numberOfThreads))
        else:
            return gridRagStacked2DHdf5Impl(labels,
                                            numberOfLabels=numberOfLabels,
                                            serialization=serialization)

if Configuration.WITH_Z5:

    def gridRagZ5(labels, numberOfLabels, blockShape=None, numberOfThreads=-1):

        # for z5 files, we pass a tuple containing the path to the file and the dataset key
        assert len(labels) == 2
        labelPath, labelKey = labels

        # TODO we only need this, because we cannot link properly to the z5 python bindings
        # TODO support more dtypes
        labelWrapper = nifty.z5.datasetWrapper('uint32', os.path.join(labelPath, labelKey))
        dim = len(labelWrapper.shape)
        blockShape_ = [100] * dim if blockShape is None else blockShape

        if dim == 2:
            return gridRag2DZ5(labelWrapper,
                               numberOfLabels=numberOfLabels,
                               blockShape=blockShape_,
                               numberOfThreads=int(numberOfThreads))
        elif dim == 3:
            return gridRag3DZ5(labelWrapper,
                               numberOfLabels=numberOfLabels,
                               blockShape=blockShape_,
                               numberOfThreads=int(numberOfThreads))
        else:
            raise RuntimeError("gridRagZ5 is only implemented for 2D and 3D not for %dD" % dim)


    def gridRagStacked2DZ5(labels, numberOfLabels, numberOfThreads=-1, serialization=None):

        # for z5 files, we pass a tuple containing the path to the file and the dataset key
        assert len(labels) == 2
        labelPath, labelKey = labels

        # TODO we only need this, because we cannot link properly to the z5 python bindings
        # TODO support more dtypes
        labelWrapper = nifty.z5.DatasetWrapperUint32(os.path.join(labelPath, labelKey))
        dim = len(labelWrapper.shape)
        assert dim == 3, "Stacked rag is only available for 3D labels"

        if serialization is None:
            return gridRagStacked2DZ5Impl(labelWrapper,
                                          numberOfLabels=numberOfLabels,
                                          numberOfThreads=int(numberOfThreads))
        else:
            return gridRagStacked2DZ5Impl(labelWrapper,
                                          numberOfLabels=numberOfLabels,
                                          serialization=serialization)


if WITH_H5PY:

    # TODO write the type of the rag
    def writeStackedRagToHdf5(rag, savePath):
        with h5py.File(savePath) as f:
            f.create_dataset('numberOfNodes', data=rag.numberOfNodes)
            f.create_dataset('numberOfEdges', data=rag.numberOfEdges)
            f.create_dataset('uvIds', data=rag.uvIds())
            f.create_dataset('minMaxLabelPerSlice', data=rag.minMaxLabelPerSlice())
            f.create_dataset('numberOfNodesPerSlice', data=rag.numberOfNodesPerSlice())
            f.create_dataset('numberOfInSliceEdges', data=rag.numberOfInSliceEdges())
            f.create_dataset('numberOfInBetweenSliceEdges',
                             data=rag.numberOfInBetweenSliceEdges())
            f.create_dataset('inSliceEdgeOffset', data=rag.inSliceEdgeOffset())
            f.create_dataset('betweenSliceEdgeOffset', data=rag.betweenSliceEdgeOffset())
            f.create_dataset('totalNumberOfInSliceEdges', data=rag.totalNumberOfInSliceEdges)
            f.create_dataset('totalNumberOfInBetweenSliceEdges',
                             data=rag.totalNumberOfInBetweenSliceEdges)
            f.create_dataset('edgeLengths', data=rag.edgeLengths())

    # TODO read the type of the rag
    def readStackedRagFromHdf5(labels, numberOfLabels, savePath):
        assert labels.ndim == 3, "Stacked rag is only available for 3D labels"
        serialization = []
        # load the serialization from h5
        with h5py.File(savePath, 'r') as f:
            # serialization of the undirected graph
            serialization.append(numpy.array(f['numberOfNodes'][:], dtype='uint64'))
            serialization.append(numpy.array(f['numberOfEdges'][:], dtype='uint64'))
            serialization.append(f['uvIds'][:].ravel().astype('uint64', copy=False))

            # serialization of the stacked rag
            serialization.append(numpy.array(f['totalNumberOfInSliceEdges'][:], dtype='uint64'))
            serialization.append(numpy.array(f['totalNumberOfInBetweenSliceEdges'][:],
                                             dtype='uint64'))

            # load all the per slice data to squeeze it in the format we need for serializing
            # cf. nifty/include/nifty/graph/rag/grid_rag_stacked_2d.hxx serialize
            inSliceDataKeys = ['numberOfNodesPerSlice',
                               'numberOfInSliceEdges',
                               'numberOfInBetweenSliceEdges',
                               'inSliceEdgeOffset',
                               'betweenSliceEdgeOffset']
            perSliceData = numpy.concatenate([f[key][:, None] for key in inSliceDataKeys], axis=1)
            perSliceData = numpy.concatenate([perSliceData, f['minMaxLabelPerSlice'][:]], axis=1)
            serialization.append(perSliceData.ravel().astype('uint64', copy=False))
            serialization.append(f['edgeLengths'][:].astype('uint64', copy=False))

        # get the rag from serialization + labels
        serialization = numpy.array(serialization, dtype='uint64')
        return gridRagStacked2DHdf5Impl(labels,
                                        numberOfLabels=numberOfLabels,
                                        serialization=serialization)
