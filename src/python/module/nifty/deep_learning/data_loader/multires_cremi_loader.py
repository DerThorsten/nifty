from __future__ import division,print_function
import h5py
import numpy
import os
import logging
import random 
import skimage
import skimage.transform
import sys

from ...tools import rand_choice,rand_from_range_list




def resize_2d(raw, shape, order):

    out_shape =  shape[0], shape[1], raw.shape[2]
    out = numpy.zeros(out_shape, dtype='float32')

    for z in range(raw.shape[2]):
        raw_z = raw[:,:, z]

        out[:,:,z] = skimage.transform.resize(raw_z, shape, 
            order=order, preserve_range=True, mode='reflect')
    return out


class PaddedCremiDset(object):

    def __init__(self, filename, letter, index):
        """
        @brief   Class for cremi dataset instance.
        Represents either samplem A,B, or C in the padded version
        """
        assert os.path.isfile(filename) 
        self.letter = letter
        self.index = index
        self.h5_file = h5py.File(filename,'r')

        self.raw        = self.h5_file['volumes/raw']
        self.neuron_ids = self.h5_file['volumes/labels/neuron_ids']
        self.clefts = self.h5_file['volumes/labels/clefts']

        # begin and end of the label volume
        offset = self.h5_file['annotations'].attrs['offset']
        offset = offset[0]/40,offset[1]/4,offset[2]/4
        self.labels_begin = numpy.array(offset, dtype='uint32')
        self.labels_end   = self.labels_begin + self.neuron_ids.shape

        overlap = [ (rs-ns) for  rs,ns in zip(self.raw.shape, self.neuron_ids.shape)]


    def get_padded(self, core_size, padding, z_range):
        nid_shape = self.neuron_ids.shape

        # get random core begin coordinates'
        # in them coordinate space of the labels
        nid_begin = [ random.randint(0, nid_shape[i]-core_size)for i in [1,2]]
        nid_end   = [ b + core_size for b in nid_begin]

        logging.debug('    z: %s'%str(z_range))
        logging.debug('    neuron_ids: %s %s'%(nid_begin, nid_end))

        # get the neuron_ids
        neuron_ids = self.neuron_ids[z_range[0]: z_range[1], 
            nid_begin[0]:nid_end[0],
            nid_begin[1]:nid_end[1]]

        # get the clefts
        clefts = self.clefts[z_range[0]: z_range[1], 
            nid_begin[0]:nid_end[0],
            nid_begin[1]:nid_end[1]]



        # get the coordinates of the padded raw data
        # in the coordinate space of the raw data
        offset = self.labels_begin[1:3]
        raw_begin = [ (b+o) - padding for b,o in zip(nid_begin, offset) ]
        raw_end   = [ (e+o) + padding for e,o in zip(nid_end  , offset) ]

        oz = self.labels_begin[0]
        logging.debug('    raw     : %s %s'%(raw_begin, raw_end))
        raw = self.raw[z_range[0] + oz: z_range[1] + oz ,
            raw_begin[0]:raw_end[0],
            raw_begin[1]:raw_end[1]]


        # moveaxis
        raw =        numpy.moveaxis(raw,0,2)
        neuron_ids = numpy.moveaxis(neuron_ids,0,2)
        clefts =     numpy.moveaxis(clefts,0,2)

        return raw,neuron_ids,clefts

class MultiresCremiLoader(object):

    def __init__(self,  root_folder, **kwargs):

        logging.debug('MultiresLoCremiader.__init__')

        # remembers all kwargs
        self.kwargs = kwargs

        # open all datasets
        self.dsets = []
        for i,l in enumerate(['A', 'B', 'C']):

            filename = os.path.join(root_folder, "sample_%s_padded_20160501.hdf"%l)
            logging.debug('     Load' + filename)
            ds = PaddedCremiDset(filename=filename, letter=l, index=i)
            self.dsets.append(ds)

        # dset probs
        self.dset_probs = kwargs.get('dset_probs',[1.0]*3)


    def _alloc_(self, batch_size):

        patch_sizes = self.kwargs.get('patch_sizes')
        n_resolutions = len(patch_sizes)

        raw_per_res_batch         = [None]*n_resolutions
        neuron_ids_per_pes_batch = [None]*1
        clefts_per_res_batch      = [None]*1

        # alloc
        for ri in range(n_resolutions):
            raw_shape = [patch_sizes[ri]]*2
            labels_shape = [patch_sizes[0]//(2**ri)]*2
            labels_batch_shape = [batch_size] + labels_shape + [ self.kwargs['n_z']]
            raw_batch_shape = [batch_size] + list(raw_shape) + [ self.kwargs['n_z']]

            raw_per_res_batch[ri]        = numpy.zeros(raw_batch_shape)
            if ri == 0:
                neuron_ids_per_pes_batch[ri] = numpy.zeros(labels_batch_shape)
                clefts_per_res_batch[ri]     = numpy.zeros(labels_batch_shape)

        return  raw_per_res_batch, neuron_ids_per_pes_batch, clefts_per_res_batch


    # get data
    def __call__(self, batch_size = 1):
        logging.debug('MultiresLoCremiader.__call__')

        patch_sizes = self.kwargs.get('patch_sizes')
        n_resolutions = len(patch_sizes)

        # alloc
        raw_per_res_batch, neuron_ids_per_pes_batch, clefts_per_res_batch = self._alloc_(batch_size)

        for bi in range(batch_size):  
            # get a random dataset
            dset = rand_choice(self.dset_probs, self.dsets)
            logging.debug('    dset: %s'%dset.letter)

            total_patch_size =  patch_sizes[-1] * (2**(n_resolutions-1))
            psize = (total_patch_size - patch_sizes[0])//2


            # get the padded raw and the neuron ids
            raw, neuron_ids, clefts = dset.get_padded(core_size=patch_sizes[0], padding=psize, 
                z_range= self.get_rand_z(dset))

            # make the fancy resolutions
            for r in range(n_resolutions):
                logging.debug("    "+str(r)+'/'+str(n_resolutions))
                if r == 0:
                    raw_r = raw[psize:psize+patch_sizes[0], 
                                psize:psize+patch_sizes[0],:]
                    if self.kwargs['neuron_ids']:
                        neuron_ids_per_pes_batch[r][bi,...] = neuron_ids
                    if self.kwargs['clefts']:
                        clefts_per_res_batch[r][bi,...] = clefts
                else:
                    patch_size_lr_hr = patch_sizes[r] * (2**r)
                    psize = (total_patch_size - patch_size_lr_hr)//2
                    raw_r = raw[psize:psize+patch_size_lr_hr,psize:psize+patch_size_lr_hr,:]
                    raw_r = resize_2d(raw_r, [patch_sizes[r]]*2, order=1)

                    #if self.kwargs['neuron_ids']:
                    #    neuron_ids_per_pes_batch[r][bi,...] = neuron_ids_per_pes_batch[r-1][bi,::2,::2,:]
                    #if self.kwargs['clefts']:
                    #    clefts_per_res_batch[r][bi,...] = clefts_per_res_batch[r-1][bi,::2,::2,:]
                raw_per_res_batch[r][bi,...] = raw_r

        if self.kwargs['neuron_ids'] and  self.kwargs['clefts']:
            return raw_per_res_batch, neuron_ids_per_pes_batch[0],clefts_per_res_batch[0]
        if not self.kwargs['neuron_ids'] and  self.kwargs['clefts']:
            return raw_per_res_batch,clefts_per_res_batch[0]
        if self.kwargs['neuron_ids'] and  not self.kwargs['clefts']:
            return raw_per_res_batch, neuron_ids_per_pes_batch[0]


 



    def get_rand_z(self, dset):        
        oz = dset.labels_begin[0]
        rslice_ranges = self.kwargs['z_slice_ranges']  
        bad_slices = self.kwargs['z_slice_ranges'][dset.index]
        z_begin = rand_from_range_list(rslice_ranges, bad_slices)
        return z_begin  , z_begin + self.kwargs['n_z']

if __name__ == "__main__":

    #logging.basicConfig(filename='example.log',level=logging.DEBUG)
    logging.basicConfig(level=logging.DEBUG)

    kwargs = {
        'neuron_ids' : True,
        'clefts' :     True,
        'patch_sizes' : [200,200,200],
        'n_z' : 5,
        'z_slice_ranges' : [[0,15 ], [50,55], [110,120]],
        'bad_slices'     : [[],[],[14]]

    }
    root_folder="/media/tbeier/4D11469249B96057/work/cremi/data/raw_padded/"
    loader = MultiresLoader(root_folder=root_foCremilder, **kwargs)


    raw_list, neuron_ids, clefts = loader(batch_size=10)

    print(clefts[0].shape)