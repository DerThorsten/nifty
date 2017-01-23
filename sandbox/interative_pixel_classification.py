import nifty
import nifty.pipelines
print nifty.__file__

import nifty.hdf5
import nifty.pipelines.ilastik_backend as ilastik_backend


def makeH5InputFile(filename, dataset, hashTabelSize=977,
                    nBytes=36000000, rddc=1.0):

    cacheSettings = nifty.hdf5.CacheSettings(hashTabelSize=hashTabelSize,
                                             nBytes=nBytes, rddc=rddc)
    h5File = nifty.hdf5.openFile(filename)
    h5Array = nifty.hdf5.Hdf5ArrayUInt8(h5File, str(dataset))
    inputFile = ilastik_backend.hdf5InputUInt32Float3D(h5Array)
    return inputFile



# the class
Ipc = ilastik_backend.InteractivePixelClassificationSpatial3D
ipc = Ipc()


filename = "/home/tbeier/Desktop/data_normalized_SUBSAMPLED.h5"
inputFile = makeH5InputFile(filename, 'data')


ipc.addTrainingInstance(inputFile)

