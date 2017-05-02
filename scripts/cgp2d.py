from  __future__ import print_function,division

import nifty
import nifty.cgp as ncgp
import h5py
import numpy
import vigra

f = "/home/tbeier/Downloads/sample_A_20160501.hdf"
f = h5py.File(f)
raw = f['volumes/raw'][0,:,:].astype('float32')[0:500,0:500].squeeze()
gt  = f['volumes/labels/neuron_ids'][0,:,:].astype('uint32')[0:500,0:500]
#print(gt.min())

def watersheds(raw, sigma):
    edgeIndicator = vigra.filters.hessianOfGaussianEigenvalues(raw, sigma)[:,:,0]
    seg, nseg = vigra.analysis.watersheds(edgeIndicator)
    #seg -= 1
    #vigra.segShow(raw, seg)
    #vigra.show()
    return seg,nseg





seg, nseg = watersheds(raw, 7.0)



print(nseg)

cgp = ncgp.TopologicalGrid2D(seg)
filled_tgrid = ncgp.FilledTopologicalGrid2D(cgp)

cell_1_img = filled_tgrid.cellMask(showCells=[False,True,False])
cell_1_img[cell_1_img!=0] -= filled_tgrid.cellTypeOffset[1]

# vigra.imshow(cell_1_img)
# vigra.show()


geometry = cgp.extractCellsGeometry(fill=True, sort1Cells=True)
bounds = cgp.extractCellsBounds()
bounds_0 = bounds[0]
bounds_1 = bounds[1]
bounded_by_1 = bounds_0.reverseMapping()
bounded_by_2 = bounds_1.reverseMapping()
geometry_0 = geometry[0]
geometry_1 = geometry[1]
geometry_2 = geometry[2]


features = ncgp.Cell1CurvatureFeatures2D(sigmas=[1.5],quantiles=[0.5])(
    cell1GeometryVector=geometry_1,
    cell1BoundedByVector=bounded_by_1,
)


# features = ncgp.Cell1LineSegmentDist(dists=[15])(
#     cell1GeometryVector=geometry_1
# )

print("lemima", features.min(),features.max())










import matplotlib
cm = matplotlib.cm.ScalarMappable(cmap='plasma')
rgb = cm.to_rgba(features[:,0])[:,0:3]
print("lutshape",rgb.shape, cgp.numberOfCells[1])

shape = cell_1_img.shape + (3,)
img = ncgp.makeCellImage(numpy.zeros(shape),mask_image=cell_1_img, lut=rgb)
print(features)
vigra.imshow(img)
vigra.show()