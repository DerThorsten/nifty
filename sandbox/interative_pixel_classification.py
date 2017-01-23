import nifty
import nifty.pipelines
print nifty.__file__

import nifty.hdf5
import nifty.pipelines.ilastik_backend as ilastik_backend




# the class
Ipc = ilastik_backend.InteractivePixelClassificationSpatial3D
ipc = Ipc()


# add the raw data
filePath = "/home/tbeier/Desktop/data/smallFibStack.h5"
cacheSettings = nifty.hdf5.CacheSettings(hashTabelSize=977,
                                         nBytes=36000000, rddc=1.0)
h5File = nifty.hdf5.openFile(filePath)


