import numpy
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


# training instance
filename = "/home/tbeier/Desktop/data_normalized_SUBSAMPLED.h5"
inputFile = makeH5InputFile(filename, 'data')
instanceIndex = ipc.addTrainingInstance(inputFile)


# the class
Ipc = ilastik_backend.InteractivePixelClassificationSpatial3D
ipc = Ipc(instanceIndex, 2, [64, 64, 64])




# add a slice of labels
labels = numpy.zeros([10,10,1])
labels[5,:,0] = 1
labels[8,:,0] = 1
begin = (10,10, 10)
end =  (10,10, 1)
ipc.addTrainingData(instanceIndex=instanceIndex, labels=labels, begin=begin,  end=end)