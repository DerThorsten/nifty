from __future__ import absolute_import
from __future__ import print_function

from ._nifty import *




import types
from functools import partial
import numpy
import time
import sys

from . import graph
from . import ground_truth
from . import tools
from . import ufd

from . import hdf5
#from . import marray
#from . import deep_learning


class Timer:
    def __init__(self, name=None, verbose=True):
        """
        @brief      Class for timer.
        """
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.dt = self.end - self.start
        if self.verbose:
            if self.name is not None:
                print(self.name,"took",self.dt,"sec")
            else:
                print("took",self.dt,"sec")



