from __future__ import absolute_import
from ._ufd import *


__all__ = []
for key in _ufd.__dict__.keys():
    __all__.append(key)



def ufd(size, dtype='uint64'):
    if dtype not in ['uint32','uint64']:
        raise RuntimeError("dtype must be 'uint32' or 'uint64'")
    if dtype == 'uint32':
        return Ufd_UInt32(long(size))
    elif dtype == 'uint64':
        return Ufd_UInt64(long(size))