import nifty
import numpy
import vigra


raw = vigra.impex.readHDF5("/home/tbeier/datasets/isbi_2013/test-input.h5",'data')
print raw.shape
nZ =raw.shape[2]
sp = []
nSegTotal = 0
for z in range(20):
    print z
    raw2d = raw[:,:,z]
    ei = vigra.filters.hessianOfGaussianEigenvalues(raw2d,1.7)[:,:,0]
    seg, nSeg = vigra.analysis.watershedsNew(ei)
    seg -= 1
    seg += nSegTotal
    nSegTotal += nSeg

    sp.append(seg[:,:,None])
seg = numpy.concatenate(sp+sp+sp+sp,axis=2).astype('uint64')




rag = nifty.graph.Rag3d2d()

rag.assignLabels(seg)

print rag
