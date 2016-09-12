import random
import os
import h5py
import warnings
import multiprocessing
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor

from superpixels import *

from colorama import init, Fore, Back, Style
init()




class ThreadPoolExecutorStackTraced(ThreadPoolExecutor):

    def submit(self, fn, *args, **kwargs):
        """Submits the wrapped function instead of `fn`"""

        return super(ThreadPoolExecutorStackTraced, self).submit(
            self._function_wrapper, fn, *args, **kwargs)

    def _function_wrapper(self, fn, *args, **kwargs):
        """Wraps `fn` in order to preserve the traceback of any kind of
        raised exception

        """
        try:
            return fn(*args, **kwargs)
        except Exception:
            raise sys.exc_info()[0](traceback.format_exc())  # Creates an
                                                             # exception of the
                                                             # same type with the
                                                             # traceback as
                                                             # message



def printWarning(txt):
    print(Fore.RED + Back.BLACK + txt)
    print(Style.RESET_ALL)

def ensureDir(f):
    if not os.path.exists(f):
        os.makedirs(f)


def hasH5File(f, key=None):
    if os.path.exists(f):
        return True
    else:
        return False

def isH5Path(h5path):
    if isinstance(h5path, tuple):
        if len(h5path) == 2:
            return True
    else:
        return False


def threadExecutor(nThreads = multiprocessing.cpu_count()):
    return ThreadPoolExecutorStackTraced(max_workers=nThreads) 



def h5Read(f, d='data'):
    f5 = h5py.File(f,'r')
    array = f5[d][:]
    f5.close()
    return array