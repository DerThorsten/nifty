from . import _distributed as __distributed
from ._distributed import *

__all__ = []
for key in __distributed.__dict__.keys():
    try:
        __distributed.__dict__[key].__module__='nifty.distributed'
    except:
        pass
    __all__.append(key)
