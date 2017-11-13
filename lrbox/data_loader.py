import h5py
import numpy
import skimage.io
import sys


def load_isbi_3d(slicing=None, mode='test'):


    offsets = numpy.array([
        [-1, 0, 0], [0, -1, 0], [0, 0, -1],                  # direct 3d nhood for attractive edges
        [-1, -1, -1], [-1, 1, 1], [-1, -1, 1], [-1, 1, -1],  # indirect 3d nhood for dam edges
        [0, -9, 0], [0, 0, -9],                  # long range direct hood
        [0, -9, -9], [0, 9, -9], [0, -9, -4], [0, -4, -9], [0, 4, -9], [0, 9, -4],  # inplane diagonal dam edges
        [0, -27, 0], [0, 0, -27]
    ]).astype('int64')

    n_offsets = offsets.shape[0]

    # the test data
    affF = "/home/tbeier/nice_probs/isbi_%s_offsetsV4_3d_meantda_damws2deval_final.h5"%mode
    affF = h5py.File(affF)

    if slicing is not None:
        s = slicing
        aff = affF['data'][
            :, s[0].start: s[0].stop,
               s[1].start: s[1].stop,
               s[2].start: s[2].stop
        ]
    else:
        aff = affF['data'][:,:,:,:]
    affF.close()

    aff = numpy.rollaxis(aff,0, 4)

    # load raw
    import skimage.io
    raw_path = "/home/tbeier/src/nifty/src/python/examples/multicut/NaturePaperDataUpl/ISBI2012/raw_%s.tif"%mode
    #raw_path = '/home/tbeier/src/nifty/mysandbox/NaturePaperDataUpl/ISBI2012/raw_test.tif'
    raw = skimage.io.imread(raw_path)
    print(raw.shape)
    if slicing is not None:
        raw = raw[slicing]

    return aff, offsets, raw



def load_predcomuted(slicing=None, mode='test'):

    filenames = [
        "/home/tbeier/nice_probs/mst_%s_offsetsV4_tdamean_damwseval.h5"%mode,
        "/home/tbeier/nice_probs/%s_aggl_mean_delayed_lifted.h5"%mode,
        "/home/tbeier/nice_probs/%s_aggl_median_delayed_lifted.h5"%mode
    ]

    segs = []
    for filename in filenames:
        h5_file = h5py.File(filename, 'r')
        ds = h5_file['data']    

        if slicing is not None:
            s = slicing
            seg = ds[
                s[0].start: s[0].stop,
                s[1].start: s[1].stop,
                s[2].start: s[2].stop
            ]
            seg = numpy.require(seg, dtype='uint64')
        else:
            seg = ds[:,:,:]

        segs.append(seg)

    return segs
