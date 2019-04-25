from __future__ import absolute_import
from __future__ import print_function

from . import _tools as __tools
from ._tools import *

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
import time
import sys
import threading
import random
import numpy


try:
    import progressbar as _progressbar
    _hasProgressbar = True
except ImportError:
    _hasProgressbar = False




__all__ = []

for key in __tools.__dict__.keys():
    __all__.append(key)
    try:
        __tools.__dict__[key].__module__='nifty.tools'
    except:
        pass





def rand_from_range_list(range_list, bad_slice_set):
    while(True):
        i = random.randint(0, len(range_list)-1)
        r = range_list[i]
        res =  random.randint(r[0],r[1])
        if res not in bad_slice_set:
            return res


# stupid impl....
# TODO make better
def rand_choice(probs, choose_from=None):
    probs = numpy.require(probs, dtype='float32')
    probs /= probs.sum()
    if choose_from is None:
       choose_from = range(len(probs))
    return numpy.random.choice(choose_from, 1, p=probs)[0]




def take(relabeling, toRelabel):
    shape = toRelabel.shape
    toRelabelFlat = toRelabel.ravel()
    return _tools._take(relabeling, toRelabelFlat).reshape(shape)



def takeDict(relabeling, toRelabel):
    shape = toRelabel.shape
    toRelabelFlat = toRelabel.ravel()
    return _tools._takeDict(relabeling, toRelabelFlat).reshape(shape)



def getSlicing(begin, end):
    return [slice(b,e) for b,e in zip(begin,end)]


def blocking(roiBegin, roiEnd, blockShape, blockShift=None):
    ndim = len(roiBegin)

    assert ndim == len(roiEnd)
    assert ndim == len(blockShape)
    if blockShift is not None:
        assert ndim == len(blockShift)
    else:
        blockShift = [0]*ndim


    if ndim == 1:
        blockingCls = Blocking1d
    elif ndim == 2:
        blockingCls = Blocking2d
    elif ndim == 3:
        blockingCls = Blocking3d
    else:
        raise RuntimeError("only 2d and 3d blocking is implemented currently")

    return blockingCls(
        [int(v) for v in roiBegin],
        [int(v) for v in roiEnd],
        [int(v) for v in blockShape],
        [int(v) for v in blockShift]
    )



def parallelForEach(iterable, f, nWorkers=cpu_count() ,
                    showBar=False, size=None, name=None):
    if nWorkers == -1 or nWorkers is None:
        nWorkers = cpu_count()
    if not showBar:
        if nWorkers == 1 or nWorkers == 0:
            for i in iterable:
                f(i)
        else:
            with ThreadPoolExecutor(max_workers=nWorkers) as e:
                for i in iterable:
                    e.submit(f,i)

    else:
        if size is None:
            raise RuntimeError("if showBar==True, size must be specified")

        lock = threading.Lock()
        done = [0]

        with progressBar(size=size, name=name) as bar:
            def fTilde(val):

                f(val)

                with lock:
                    done[0] += 1
                    bar.update(done[0])
            parallelForEach(iterable=iterable,
                            f=fTilde,
                            nWorkers=nWorkers,
                            showBar=False,
                            size=None)





if not _hasProgressbar:

    class Progressbar:
        def __init__(self, maxValue, name=""):
            self.maxValue = maxValue

        def __enter__(self):
            #set things up
            return self

        def __exit__(self, type, value, traceback):

            infoStr = str(self.maxValue) + '/' + str(self.maxValue)
            sys.stdout.write('%s' % infoStr.ljust(20))
            pass
        def update(self, val):
            infoStr = str(val) + '/' + str(self.maxValue)
            sys.stdout.write('%s\r' % infoStr.ljust(20))
            sys.stdout.flush()
else:
    class Progressbar:
        def __init__(self, size, name=None):
            if name is None:
                name = ""
            widgets = [

                ' [',str(name), _progressbar.Timer(), ', ',_progressbar.Counter(),'/%s'%size,'] ',
                 _progressbar.Bar(),
                ' (', _progressbar.ETA(), ') ',
            ]

            self.bar = _progressbar.ProgressBar(
                maxval=size, widgets=widgets
            )

        def __enter__(self):
            self.bar.start()
            return self

        def __exit__(self, type, value, traceback):
            self.bar.finish()
            pass
        def update(self, val):
            self.bar.update(val)


def progressBar(size, name=None):
    return Progressbar(size,name)
