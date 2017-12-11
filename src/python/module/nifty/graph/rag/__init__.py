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


def gridRag(labels,
            numberOfLabels,
            blockShape=None,
            numberOfThreads=-1,
            serialization=None,
            dtype='uint32'):
    labels = numpy.require(labels, dtype=dtype)
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
        factory = explicitLabelsGridRag3D32 if dtype == numpy.dtype('uint32') \
            else explicitLabelsGridRag3D64
        if serialization is None:
            return factory(labels,
                           blockShape=blockShape_,
                           numberOfLabels=numberOfLabels,
                           numberOfThreads=int(numberOfThreads))
        else:
            return factory(labelsProxy,
                           numberOfLabels=numberOfLabels,
                           serialization=serialization)

    else:
        raise RuntimeError("wrong dimension, currently only 2D and 3D is implemented")


def gridRagStacked2D(labels,
                     numberOfLabels,
                     serialization=None,
                     numberOfThreads=-1,
                     dtype='uint32'):
    labels = numpy.require(labels, dtype=dtype)
    assert labels.ndim == 3, "Stacked rag is only available for 3D labels"
    factory = gridRagStacked2D32 if dtype == numpy.dtype('uint32') \
        else gridRagStacked2D64
    if serialization is None:
        return factory(labels,
                       numberOfLabels=numberOfLabels,
                       numberOfThreads=numberOfThreads)
    else:
        return factory(labels,
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

    def gridRagHdf5(labels,
                    numberOfLabels,
                    blockShape=None,
                    numberOfThreads=-1,
                    dtype='uint32'):

        dim = labels.ndim
        blockShape_ = [100] * dim if blockShape is None else blockShape

        if dim == 2:
            return gridRag2DHdf5(labels,
                                 numberOfLabels=numberOfLabels,
                                 blockShape=blockShape_,
                                 numberOfThreads=int(numberOfThreads))
        elif dim == 3:
            factory = gridRag3DHdf532 if dtype == numpy.dtype('uint32') \
                else gridRag3DHdf564
            return factory(labels,
                           numberOfLabels=numberOfLabels,
                           blockShape=blockShape_,
                           numberOfThreads=int(numberOfThreads))
        else:
            raise RuntimeError("gridRagHdf5 is only implemented for 2D and 3D not for %dD" % dim)


    def gridRagStacked2DHdf5(labels,
                             numberOfLabels,
                             numberOfThreads=-1,
                             serialization=None,
                             dtype='uint32'):
        assert labels.ndim == 3, "Stacked rag is only available for 3D labels"
        factory = gridRagStacked2DHdf532 if dtype == numpy.dtype('uint32') \
            else gridRagStacked2DHdf564
        if serialization is None:
            return factory(labels,
                           numberOfLabels=numberOfLabels,
                           numberOfThreads=int(numberOfThreads))
        else:
            return factory(labels,
                           numberOfLabels=numberOfLabels,
                           serialization=serialization)

if Configuration.WITH_Z5:

    def gridRagZ5(labels,
                  numberOfLabels,
                  blockShape=None,
                  numberOfThreads=-1,
                  dtype='uint32'):

        # for z5 files, we pass a tuple containing the path to the file and the dataset key
        assert len(labels) == 2
        labelPath, labelKey = labels

        # TODO we only need this, because we cannot link properly to the z5 python bindings
        # TODO support more dtypes
        labelWrapper = nifty.z5.datasetWrapper(dtype, os.path.join(labelPath, labelKey))
        dim = len(labelWrapper.shape)
        blockShape_ = [100] * dim if blockShape is None else blockShape

        if dim == 2:
            return gridRag2DZ5(labelWrapper,
                               numberOfLabels=numberOfLabels,
                               blockShape=blockShape_,
                               numberOfThreads=int(numberOfThreads))
        elif dim == 3:
            factory = gridRag3DZ532 if dtype == numpy.dtype('uint32') \
                else gridRag3DZ564
            return factory(labelWrapper,
                           numberOfLabels=numberOfLabels,
                           blockShape=blockShape_,
                           numberOfThreads=int(numberOfThreads))
        else:
            raise RuntimeError("gridRagZ5 is only implemented for 2D and 3D not for %dD" % dim)


    def gridRagStacked2DZ5(labels,
                           numberOfLabels,
                           numberOfThreads=-1,
                           serialization=None,
                           dtype='uint32'):

        # for z5 files, we pass a tuple containing the path to the file and the dataset key
        assert len(labels) == 2
        labelPath, labelKey = labels

        # TODO we only need this, because we cannot link properly to the z5 python bindings
        labelWrapper = nifty.z5.datasetWrapper(dtype, os.path.join(labelPath, labelKey))
        dim = len(labelWrapper.shape)
        assert dim == 3, "Stacked rag is only available for 3D labels"

        factory = gridRagStacked2DZ532 if dtype == numpy.dtype('uint32') \
            else gridRagStacked2DZ564
        if serialization is None:
            return factory(labelWrapper,
                           numberOfLabels=numberOfLabels,
                           numberOfThreads=int(numberOfThreads))
        else:
            return factory(labelWrapper,
                           numberOfLabels=numberOfLabels,
                           serialization=serialization)


if WITH_H5PY:

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

            className = rag.__class__.__name__

            if className.startswith('GridRagStacked2DZ5'):
                factory = 'gridRagStacked2DZ5'
            elif className.startswith('GridRagStacked2DHdf5'):
                factory = 'gridRagStacked2DHdf5'
            elif className.startswith('GridRagStacked2D'):
                factory = 'gridRagStacked2D'

            # write factory and dtype as attributes
            attrs = f.attrs
            attrs['factory'] = factory
            dtype = 'uint' + className[-2:]
            attrs['dtype'] = dtype

    def readStackedRagFromHdf5(labels, numberOfLabels, savePath):
        serialization = []
        # load the serialization from h5
        with h5py.File(savePath, 'r') as f:
            # serialization of the undirected graph
            eAndN = numpy.zeros((2, 1), dtype='uint64')
            eAndN[0] = f['numberOfNodes']
            eAndN[1] = f['numberOfEdges']
            serialization.append(eAndN)
            serialization.append(f['uvIds'][:].ravel().astype('uint64', copy=False)[:, None])

            # serialization of the stacked rag
            tEdges = numpy.zeros((2, 1), dtype='uint64')
            tEdges[0] = f['totalNumberOfInSliceEdges']
            tEdges[1] = f['totalNumberOfInBetweenSliceEdges']
            serialization.append(tEdges)

            # load all the per slice data to squeeze it in the format we need for serializing
            # cf. nifty/include/nifty/graph/rag/grid_rag_stacked_2d.hxx serialize
            inSliceDataKeys = ['numberOfInSliceEdges',
                               'numberOfInBetweenSliceEdges',
                               'inSliceEdgeOffset',
                               'betweenSliceEdgeOffset']
            perSliceData = numpy.concatenate([numpy.array(f[key], dtype='uint64')[:, None]
                                              for key in inSliceDataKeys], axis=0)
            minmaxLabel = f['minMaxLabelPerSlice'][:]
            perSliceData = numpy.concatenate([perSliceData, minmaxLabel[:, :1], minmaxLabel[:, 1:]],
                                             axis=0)
            serialization.append(perSliceData.astype('uint64', copy=False))
            serialization.append(f['edgeLengths'][:].astype('uint64', copy=False)[:, None])

            # read factory and dtype from attributes
            attrs = f.attrs
            factory = eval(attrs['factory'])
            dtype = attrs['dtype']

        for ser in serialization:
            print(ser.shape)
        serialization = numpy.concatenate(serialization)

        return factory(labels,
                       dtype=dtype,
                       numberOfLabels=numberOfLabels,
                       serialization=serialization.squeeze())
