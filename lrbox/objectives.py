import numpy

import nifty.graph.opt.lifted_multicut as nlmc
import make_weights



class IsbiObjective(nlmc.PixelWiseLmcObjective):
    def __init__(self, offsets, affinities, raw):
        weights = make_weights.affinities_to_weights(affinities=affinities, offsets=offsets)
        super(IsbiObjective, self).__init__(offsets=offsets, weights=weights)

        self.raw = raw
        self.affinities = affinities

        

    def z_objective(self,z):

        
        raw = self.raw[z,...]
        no_z_offset = numpy.where(self.offsets[:,0]==0)[0]

        print(no_z_offset)

        offsets    = self.offsets[no_z_offset, 1:3]
        affinities = self.affinities[z,:,:]
        affinities = affinities[:,:,no_z_offset]
        print("self",self.affinities.shape)
        print("affshape",affinities.shape)
        print("ioffsetshap-e",offsets.shape)
        return IsbiObjective(affinities=affinities, 
            offsets=offsets, raw=raw)

