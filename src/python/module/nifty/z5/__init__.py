from __future__ import absolute_import
import numpy

from .. import Configuration

from ._z5 import *

__all__ = []

if Configuration.WITH_Z5:
    from ._z5 import *
    from . import _z5 as __z5
    for key in __z5.__dict__.keys():
        __all__.append(key)
        try:
            __z5.__dict__[key].__module__ = 'nifty.z5'
        except:
            pass
else:
    pass


if Configuration.WITH_Z5:

    # convenience wrapper around z5 dataset wrapper
    def datasetWrapper(dtype, *args, **kwargs):

        if numpy.dtype(dtype) == numpy.dtype("uint8"):
            return DatasetWrapperUint8(*args, **kwargs)
        elif numpy.dtype(dtype) == numpy.dtype("uint16"):
            return DatasetWrapperUint16(*args, **kwargs)
        elif numpy.dtype(dtype) == numpy.dtype("uint32"):
            return DatasetWrapperUint32(*args, **kwargs)
        elif numpy.dtype(dtype) == numpy.dtype("uint64"):
            return DatasetWrapperUint64(*args, **kwargs)

        elif numpy.dtype(dtype) == numpy.dtype("int8"):
            return DatasetWrapperInt8(*args, **kwargs)
        elif numpy.dtype(dtype) == numpy.dtype("int16"):
            return DatasetWrapperInt16(*args, **kwargs)
        elif numpy.dtype(dtype) == numpy.dtype("int32"):
            return DatasetWrapperInt32(*args, **kwargs)
        elif numpy.dtype(dtype) == numpy.dtype("int64"):
            return DatasetWrapperInt64(*args, **kwargs)

        elif numpy.dtype(dtype) == numpy.dtype("float32"):
            return DatasetWrapperFloat32(*args, **kwargs)
        elif numpy.dtype(dtype) == numpy.dtype("float64"):
            return DatasetWrapperFloat64(*args, **kwargs)

        else:
            raise RuntimeError("Datatype %s not supported!" % (str(dtype),))
