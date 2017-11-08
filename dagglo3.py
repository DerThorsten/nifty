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

mergec_z = [0]




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
affinities  = affinities[:,:,:,:]



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
raw = raw[:,:,:]
#raw = numpy.rollaxis(raw ,0,3)


print("raw shape", raw.shape)
print("affinities shape", affinities.shape)
print("offsets shape",offsets.shape)

isMergeEdgeOffset = numpy.zeros(offsets.shape[0], dtype='bool')
isMergeEdgeOffset[0] = True
isMergeEdgeOffset[1] = True
isMergeEdgeOffset[2] = True


#sys.exit()


def make_more_morse(prio, raw):
    #for z in range(prio.shape[0]):
    #    raw_xy = vigra.gaussianSmoothing(raw[z,...],0.2)
    #    prio_xyc = vigra.taggedView(prio[z,...],'xyc')
    #    prio[z,...] += 0.001*vigra.gaussianSmoothing(prio_xyc, 6.0)   + 0.0000001*(raw_xy[:,:,None])
    return prio


notMergePrios = affinities.copy()
#notMergePrios[notMergePrios<0.9999] = 0


if True:
    affinities = numpy.require(affinities, dtype='float32', requirements=['C'])
    mergePrios = make_more_morse(1.0 - affinities, raw)
    notMergePrios = make_more_morse(affinities, 255.0-raw)

    mergePrios[:,:,:,mergec_z] *= 0.9
    notMergePrios[:,:,:,[3,4,5,6]] *= 0.9

    with nifty.Timer("jay"):
        nodeSeg = nifty.graph.agglo.pixelWiseFixation3D(
            mergePrios=    mergePrios,  #vigra.gaussianSmoothing(vigra.taggedView( (1.0 - affinities),'xyc'),0.01),
            notMergePrios= notMergePrios,               #vigra.gaussianSmoothing(vigra.taggedView( (affinities),'xyc'),0.01),
            offsets=offsets,
            isMergeEdgeOffset=isMergeEdgeOffset
        )

    f = h5py.File("/home/tbeier/nice_probs/agglo_res8.h",'w')
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


