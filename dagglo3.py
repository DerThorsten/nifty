import nifty
import nifty.graph.agglo
import nifty.segmentation
import h5py
import numpy
import pylab
import vigra

# load affinities
w = 500
path_affinities = "/home/tbeier/nice_probs/isbi_test_offsetsV4_3d_meantda.h5"
offsets = numpy.array([
[-1, 0, 0], [0, -1, 0], [0, 0, -1],                  # direct 3d nhood for attractive edges
[-1, -1, -1], [-1, 1, 1], [-1, -1, 1], [-1, 1, -1],  # indirect 3d nhood for dam edges
[0, -9, 0], [0, 0, -9],                  # long range direct hood
[0, -9, -9], [0, 9, -9], [0, -9, -4], [0, -4, -9], [0, 4, -9], [0, 9, -4],  # inplane diagonal dam edges
[0, -27, 0], [0, 0, -27]
]).astype('int64')

# z = offsets[:,0].copy()
# x = offsets[:,1].copy()
# y = offsets[:,2].copy()
# #
# #
# offsets[:,0] = xz
# offsets[:,1] = y
# offsets[:,2] = z

print(len)

for offset in offsets:
    print(offset)
print(offsets.shape)

f5_affinities = h5py.File(path_affinities)
affinities = f5_affinities['data']

print(affinities.shape)

affinities = affinities[:,:,:, :]
print("in aff shape",affinities.shape)
affinities = numpy.rollaxis(affinities ,0,4)
affinities  = affinities[0:5,0:w,0:w,:]



#affinities = numpy.rollaxis(affinities ,0,3)
print(affinities.shape)
affinities = numpy.require(affinities, dtype='float32', requirements=['C'])

print(affinities.min(), affinities.max())

# pylab.imshow(affinities[0,:,:,1])
# pylab.show()


# load raw
import skimage.io
raw_path = "/home/tbeier/src/nifty/src/python/examples/multicut/NaturePaperDataUpl/ISBI2012/raw_test.tif"
#raw_path = '/home/tbeier/src/nifty/mysandbox/NaturePaperDataUpl/ISBI2012/raw_test.tif'
raw = skimage.io.imread(raw_path)
raw = raw[0:5,0:w,0:w]
#raw = numpy.rollaxis(raw ,0,3)


print("raw shape", raw.shape)
print("affinities shape", affinities.shape)
print("offsets shape",offsets.shape)

isMergeEdgeOffset = numpy.zeros(offsets.shape[0], dtype='bool')
isMergeEdgeOffset[0] = True
isMergeEdgeOffset[1] = True
isMergeEdgeOffset[2] = True


#sys.exit()



notMergePrios = affinities.copy()
#notMergePrios[notMergePrios<0.9999] = 0


if True:
    affinities = numpy.require(affinities, dtype='float32', requirements=['C'])
    with nifty.Timer("jay"):
        nodeSeg = nifty.graph.agglo.pixelWiseFixation3D(
            mergePrios=    (1.0 - affinities)+0.0,  #vigra.gaussianSmoothing(vigra.taggedView( (1.0 - affinities),'xyc'),0.01),
            notMergePrios= notMergePrios,               #vigra.gaussianSmoothing(vigra.taggedView( (affinities),'xyc'),0.01),
            offsets=offsets,
            isMergeEdgeOffset=isMergeEdgeOffset
        )

    f = h5py.File("/home/tbeier/nice_probs/agglo_res3.h5",'w')
    f['data'] = nodeSeg
    f.close()
    #nodeSeg = nodeSeg.reshape(shape)

    import pylab
    print(nodeSeg.shape)
    #pylab.imshow(nodeSeg[:,:,0])
    #pylab.show()
    pylab.imshow(nifty.segmentation.segmentOverlay(raw[3,:,:], nodeSeg[3,:,:], showBoundaries=False))
    pylab.show()
    # pylab.imshow(nifty.segmentation.markBoundaries(raw, nodeSeg, color=(1,0,0)))
    # pylab.show()


