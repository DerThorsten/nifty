from . import _skeletons as __skeletons
from ._skeletons import *

__all__ = []
for key in __skeletons.__dict__.keys():
    try:
        __skeletons.__dict__[key].__module__='nifty.skeletons'
    except:
        pass
    __all__.append(key)
