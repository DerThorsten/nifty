import nifty
import numpy
import nifty.segmentation
import nifty.graph.rag
import nifty.graph.agglo
import vigra
import matplotlib.pyplot as plt
from random import shuffle





import skimage.io
import h5py

z = 8
w = 100
# load weighs and raw
path_affinities = "/home/tbeier/nice_probs/isbi_test_default.h5"
offsets = numpy.array([
    [-1,0],[0,-1],
    [-9,0],[0,-9],[-9,-9],[9,-9],
    [-9,-4],[-4,-9],[4,-9],[9,-4],
    [-27,0],[0,-27],[-27,-27],[27,-27]
])

f5_affinities = h5py.File(path_affinities)
affinities = f5_affinities['data']
affinities = numpy.rollaxis(affinities[: , z, 0:w, 0:w],0,3)
affinities = numpy.require(affinities, requirements=['C'])

raw_path = "/home/tbeier/src/nifty/src/python/examples/multicut/NaturePaperDataUpl/ISBI2012/raw_test.tif"
raw = skimage.io.imread(raw_path)
raw = raw[z, 0:w, 0:w]



print(raw.shape, affinities.shape)





