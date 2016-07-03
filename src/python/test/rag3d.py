from __future__ import print_function
import nifty
import numpy
import h5py
import vigra






if False:
    f = "/home/tbeier/datasets/hhess/data_normalized.h5"
    hfile = h5py.File(f)
    data = hfile['data'][0:500,0:1000,0:1000].astype('float32')
    #data = 255.0 - data
    data = vigra.taggedView(data, 'xyz')
    with vigra.Timer("ew"):
        ev = vigra.filters.hessianOfGaussianEigenvalues(data,1.5)[:,:,:,0]
    with vigra.Timer("ws"):
        seg, nseg = vigra.analysis.unionFindWatershed3D(ev, (100,100,100))
        vigra.impex.writeHDF5(seg, "/home/tbeier/datasets/hhess/labels.h5", "data")
else:
    with vigra.Timer("read seg"):
        seg = vigra.impex.readHDF5("/home/tbeier/datasets/hhess/labels.h5", "data")

print("compute rag")
g =  nifty.graph.rag.explicitLabelsGridRag3D(seg)
