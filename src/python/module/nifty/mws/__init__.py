from . import _mws as __mws
from ._mws import *

__all__ = []
for key in __mws.__dict__.keys():
    try:
        __mws.__dict__[key].__module__='nifty.mws'
    except:
        pass
    __all__.append(key)
