from __future__ import absolute_import
from __future__ import print_function

from ._nifty import *




import types
import numpy
import time
import sys

from . import graph
from . import tools
from . import ufd



class Timer:
    def __init__(self, name=None, verbose=True):
        """Timer class as with statement
        
        Time pieces of code with a with statement timer

        Examples:

            import nifty
            with nifty.Timer() as t:
                import time
                time.sleep()

            
        Args:
            name: name to print (default: {None})
            verbose: do printout (default: {True})
        """
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.elapsedTime = self.end - self.start
        if self.verbose:
            if self.name is not None:
                print(self.name,"took",self.elapsedTime,"sec")
            else:
                print("took",self.elapsedTime,"sec")



