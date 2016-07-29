from __future__ import print_function
import nifty
import numpy
import h5py
import vigra






if False:
    f = "/media/tbeier/AD9A-CE7E/data_sub.h5"
    hfile = h5py.File(f)
    data = hfile['data'][0:500,0:1000,0:1000].astype('float32')
    #data = 255.0 - data
    data = vigra.taggedView(data, 'xyz')
    with vigra.Timer("ew"):
        opt = vigra.blockwise.convolutionOptions(sigma=1.5, blockShape=(100,100,100))
        ev = vigra.blockwise.hessianOfGaussianEigenvalues(data,opt)[:,:,:,0]
    with vigra.Timer("ws"):
        seg, nseg = vigra.analysis.unionFindWatershed3D(ev, (300,300,300))
        seg = numpy.array(seg-1)
        print(seg.strides)

        vigra.impex.writeHDF5(seg, "/home/tbeier/labels.h5", "data")
        vigra.segShow(data[:,:,0],seg[:,:,0])
        vigra.show()

else:
    with vigra.Timer("read seg"):
        seg = vigra.impex.readHDF5("/home/tbeier/labels.h5", "data")
        print("loaded strides", seg.strides)


with vigra.Timer("rag parallel"):
    gp =  nifty.graph.rag.explicitLabelsGridRag3D(seg,-1)

with vigra.Timer("rag serial"):
    gs =  nifty.graph.rag.explicitLabelsGridRag3D(seg,0)
