from __future__ import print_function
from _nifty import *
import types
from functools import partial
import numpy
import time
import sys



try:
    hasVolumina=True
    import volumina
    from volumina.api import Viewer
except:
    hasVolumina=False



try:
    hasVigra=True
    import vigra
except:
    hasVigra=False
    

try:
    hasQt4=True
    from PyQt4.QtGui import QApplication
except:
    hasQt4=False



class Singleton:
    """
    A non-thread-safe helper class to ease implementing singletons.
    This should be used as a decorator -- not a metaclass -- to the
    class that should be a singleton.
    The decorated class can define one `__init__` function that
    takes only the `self` argument. Other than that, there are
    no restrictions that apply to the decorated class.
    To get the singleton instance, use the `Instance` method. Trying
    to use `__call__` will result in a `TypeError` being raised.
    Limitations: The decorated class cannot be inherited from.
    """

    def __init__(self, decorated):
        self._decorated = decorated

    def Instance(self):
        """
        Returns the singleton instance. Upon its first call, it creates a
        new instance of the decorated class and calls its `__init__` method.
        On all subsequent calls, the already created instance is returned.
        """
        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated()
            return self._instance

    def __call__(self):
        raise TypeError('Singletons must be accessed through `Instance()`.')

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)


if hasQt4:
    @Singleton
    class QApp:
        def __init__(self):
            self.app = QApplication(sys.argv)










if hasQt4 and hasVolumina:
    def addHocViewer(grayData=None, segData=None, title="viewer",visu=True):
        if visu :
            app = QApp.Instance().app

            v = Viewer()

            if grayData is not None:
                for name in grayData.keys():
                    data = grayData[name]
                    if hasVigra:
                        if isinstance(data, vigra.arraytypes.VigraArray):
                            v.addGrayscaleLayer(data.view(numpy.ndarray), name=name)
                        else:
                            v.addGrayscaleLayer(data, name=name)
                    else:
                        v.addGrayscaleLayer(data, name=name)

            if segData is not None:
                for name in segData.keys():
                    data = segData[name]
                    if hasVigra:
                        if isinstance(data, vigra.arraytypes.VigraArray):
                            v.addColorTableLayer(data.view(numpy.ndarray), name=name)
                        else:
                            v.addColorTableLayer(data, name=name)
                    else:
                        v.addGrayscaleLayer(data, name=name)

            v.setWindowTitle(title)
            v.showMaximized()
            app.exec_()












class Timer:    
    def __init__(self, name=None, verbose=True):
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






if Configuration.WITH_HDF5:

    def __extendHdf5():
        hdf5Arrays = [
            hdf5.Hdf5ArrayUInt8,
            hdf5.Hdf5ArrayUInt16,
            hdf5.Hdf5ArrayUInt32,
            hdf5.Hdf5ArrayUInt64,
            hdf5.Hdf5ArrayInt8,
            hdf5.Hdf5ArrayInt16,
            hdf5.Hdf5ArrayInt32,
            hdf5.Hdf5ArrayInt64,
            hdf5.Hdf5ArrayFloat32,
            hdf5.Hdf5ArrayFloat64
        ]

        def getItem(self, slicing):
            dim = self.ndim
            roiBegin = [None]*dim
            roiEnd = [None]*dim
            for d in range(dim):
                sliceObj = slicing[d]
                roiBegin[d] = int(sliceObj.start)
                roiEnd[d] = int(sliceObj.stop)
                step = sliceObj.step
                if step is not None and  step != 1:
                    raise RuntimeError("currently step must be 1 in slicing but step is %d"%sliceObj.step)

            return self.readSubarray(roiBegin, roiEnd)

        def setItem(self, slicing, value):
            asArray = numpy.require(value)

            dim = self.ndim
            roiBegin = [None]*dim
            roiEnd = [None]*dim
            shape = [None]*dim
            for d in range(dim):
                sliceObj = slicing[d]
                roiBegin[d] = int(sliceObj.start)
                roiEnd[d] = int(sliceObj.stop)
                if roiEnd[d] - roiBegin[d] != asArray.shape[d]:
                    raise RuntimeError("array to write does not match slicing shape")
                step = sliceObj.step
                if step is not None and  step != 1:
                    raise RuntimeError("currently step must be 1 in slicing but step is %d"%sliceObj.step)

            return self.writeSubarray(roiBegin, asArray)


        for array in hdf5Arrays:
            array.__getitem__ = getItem
            array.__setitem__ = setItem






    __extendHdf5()
    del __extendHdf5
