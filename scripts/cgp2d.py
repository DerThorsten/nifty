from  __future__ import print_function,division

import nifty
import nifty.cgp as ncgp
import h5py
import numpy
import fastfilters as ffilt
import vigra

f = "/home/tbeier/Downloads/sample_A_20160501.hdf"
f = h5py.File(f)
raw = f['volumes/raw'][0,:,:].astype('float32')[0:400,0:400].squeeze()
gt  = f['volumes/labels/neuron_ids'][0,:,:].astype('uint32')[0:400,0:400]
#print(gt.min())

def watersheds(raw, sigma):
    edgeIndicator = ffilt.hessianOfGaussianEigenvalues(raw, sigma)[:,:,0]
    seg, nseg = vigra.analysis.watersheds(edgeIndicator)
    #seg -= 1
    #vigra.segShow(raw, seg)
    #vigra.show()
    return seg,nseg



seg, nseg = watersheds(raw, 3.0)

print(nseg)

cgp = ncgp.TopologicalGrid2D(seg)

print(cgp.numberOfCells(0))
print(cgp.numberOfCells(1))
print(cgp.numberOfCells(2))