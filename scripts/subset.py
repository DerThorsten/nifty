import numpy
import h5py
import os
import vigra

out = "/home/tbeier/Desktop/play"


if True:
    
    pmap = "/home/tbeier/Desktop/hess-2nm-subsampled-autocontext-predictions.h5"
    pmap = h5py.File(pmap)['predictions'][0:800,0:800,0:800,:][...,0] * 255.0
    f = h5py.File(os.path.join(out,'pmap.h5'),'w')
    f.create_dataset('data',data=pmap.astype('uint8'), chunks=(64,64,64))

if False:
    data = "/home/tbeier/Desktop/data_normalized_SUBSAMPLED.h5"
    data = h5py.File(data)['data'][0:800,0:800,0:800]
    f = h5py.File(os.path.join(out,'raw.h5'),'w')
    f.create_dataset('data',data=data, chunks=(64,64,64))

if False:

    data = h5py.File(os.path.join(out,'raw.h5'),'r')['data'][:]

    convOpt = vigra.blockwise.convOpts(sigma=3.0, blockShape=[100,100,100])
    smoothed = 1.0 - vigra.blockwise.gaussianSmooth(data.astype('float32'), convOpt)

    seg, nseg = vigra.analysis.unionFindWatershed3D(smoothed, blockShape=[100,100,100])
    seg -= 1
    
    f = h5py.File(os.path.join(out,'oseg.h5'),'w')
    dset = f.create_dataset('data',data=seg, chunks=(64,64,64), compression='gzip') 
    dset.attrs['maxLabel'] = nseg - 1
    f.close()
