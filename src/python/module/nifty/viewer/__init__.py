from __future__ import absolute_import
from __future__ import division

import numpy
import types
from functools import partial
import numpy
import time
import sys
import warnings


try:
    hasVolumina=True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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





def view3D(img, show=True, cmap='Greys',title=None, singlePlain=True):

    import pylab
    import matplotlib.cm as cm

    shape = img.shape

    f = pylab.figure()

    if singlePlain:
        pylab.imshow(img[shape[0]//2,:,:],cmap=cmap)
    else:
        f.add_subplot(2, 2, 1)  
        pylab.imshow(img[shape[0]//2,:,:],cmap=cmap)
        
        f.add_subplot(2, 2, 2) 
        pylab.imshow(img[:,shape[1]//2,:],cmap=cmap)
        
        f.add_subplot(2, 2, 3) 
        pylab.imshow(img[:,:,shape[2]//2],cmap=cmap)
    
    if title is not None:
        pylab.title(title)

    if show:
        pylab.show()



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


