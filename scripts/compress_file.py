import h5py


gtF = "/home/tbeier/ilastikdata/gt_raw_73.h5"
gtFC = "/home/tbeier/ilastikdata/gt_raw_c_73.h5"
gt  = h5py.File(gtF)['exported_data']

compressedFile = h5py.File(gtFC)
ds = compressedFile.create_dataset('data', gt.shape, 'uint32', compression='lzf',data=gt,chunks=(1,100,100,100,1))