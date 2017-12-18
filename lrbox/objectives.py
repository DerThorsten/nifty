import numpy
import sys
import nifty.graph.opt.lifted_multicut as nlmc
import make_weights
import pylab

import nifty.segmentation

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
        affinities = self.affinities[z,...]
        affinities = affinities[...,no_z_offset]
        # print("self",self.affinities.shape)
        # print("affshape",affinities.shape)
        # print("ioffsetshap-e",offsets.shape)
        return IsbiObjective(affinities=affinities, 
            offsets=offsets, raw=raw)


    def sub_objective(self, begin, end):
        slicing = [slice(b,e) for b,e in zip(begin, end)]
        a_slice = slicing +[slice(0,self.offsets.shape[0])]
        affinities = self.affinities[slicing]
        raw = self.raw[slicing]
        return IsbiObjective(affinities=affinities, 
            offsets=self.offsets, raw=raw)


    def optimize_h_blockwise(self, factory, labels=None, callback=None, d=0):
        print("self.shape",self.shape)
        opt_blockwise = True
        block_shape = [s//4 for s in self.shape]


        if block_shape[0] < 20 :
            opt_blockwise = False
            


        if opt_blockwise == False:
            seg = self.optimize(factory)
            return seg


        if labels is None:  
            labels = numpy.arange(self.n_variables).reshape(self.shape)


        if self.ndim == 2:
            if block_shape is None:
                block_shape = (100, 100)

            
            blocking_a = nifty.tools.blocking(
                roiBegin=(0,0),
                roiEnd=self.shape,
                blockShape=block_shape,
                blockShift=(0,0)
            )
            blocking_b = nifty.tools.blocking(
                roiBegin=(0,0),
                roiEnd=self.shape,
                blockShape=block_shape,
                blockShift=(block_shape[0]//2,block_shape[1]//2)
            )
            blockings = [
                blocking_b,
                blocking_a
            ]


            def opt_from_block(blocking):


                blocking_labels = labels.copy().astype('uint64')
                max_label = 0
                for block_index in range(blocking.numberOfBlocks):
                    block = blocking.getBlock(block_index)
                    
                    begin = block.begin
                    end = block.end
                    #print("begin",begin,"end",end)
                    block_obj = self.sub_objective(begin=begin, end=end) 
                    seg = block_obj.optimize_h_blockwise(factory=factory, d=d+1)
                    #img = nifty.segmentation.segmentOverlay(block_obj.raw, seg, showBoundaries=True)
                    local_max = int(seg.max())
                    blocking_labels[begin[0]:end[0], begin[1]:end[1]] = seg + 1 + max_label
                    #pylab.imshow(img)
                    #pylab.show()

                    max_label = local_max + 1 + max_label
                return blocking_labels

            labels_a = opt_from_block(blockings[0])
            labels_b = opt_from_block(blockings[1])






            

            if callback is not None:
                callback(labels_a, labels_b)
            

            G = nifty.graph.UndirectedGraph
            CCObj = G.LiftedMulticutObjective 

            fm_factory = CCObj.chainedSolversFactory([
                CCObj.liftedMulticutGreedyAdditiveFactory(),
                CCObj.liftedMulticutKernighanLinFactory(),
                CCObj.fusionMoveBasedFactory()
            ])

            fm = nlmc.PixelWiseLmcConnetedComponentsFusion2D(
                objective=self.cpp_obj(), 
                solver_factory=fm_factory)

            fused = fm.fuse(labels_a, labels_b)
    


            return fused
        else:
            raise NotImplementedError("only 3D atm")



    def optimize_blockwise(self, factory, block_shape=None, labels=None, callback=None):


        if labels is None:
            labels = numpy.arange(self.n_variables).reshape(self.shape)

        if self.ndim == 2:
            if block_shape is None:
                block_shape = (100, 100)

            
            blocking_a = nifty.tools.blocking(
                roiBegin=(0,0),
                roiEnd=self.shape,
                blockShape=block_shape,
                blockShift=(0,0)
            )
            blocking_b = nifty.tools.blocking(
                roiBegin=(0,0),
                roiEnd=self.shape,
                blockShape=block_shape,
                blockShift=(block_shape[0]//2,block_shape[1]//2)
            )
            blockings = [
                blocking_b,
                blocking_a
                #blocks_b
            ]


            def opt_from_block(blocking):


                blocking_labels = labels.copy().astype('uint64')
                max_label = 0
                for block_index in range(blocking.numberOfBlocks):
                    block = blocking.getBlock(block_index)
                    
                    begin = block.begin
                    end = block.end
                    print("begin",begin,"end",end)
                    block_obj = self.sub_objective(begin=begin, end=end) 

                    
                    seg = block_obj.optimize(factory)
                    #img = nifty.segmentation.segmentOverlay(block_obj.raw, seg, showBoundaries=True)
                    local_max = int(seg.max())
                    blocking_labels[begin[0]:end[0], begin[1]:end[1]] = seg + 1 + max_label
                    #pylab.imshow(img)
                    #pylab.show()

                    max_label = local_max + 1 + max_label
                return blocking_labels

            labels_a = opt_from_block(blockings[0])
            labels_b = opt_from_block(blockings[1])

            if callback is not None:
                callback(labels_a, labels_b)
            

            G = nifty.graph.UndirectedGraph
            CCObj = G.LiftedMulticutObjective 

            fm_factory = CCObj.chainedSolversFactory([
                CCObj.liftedMulticutGreedyAdditiveFactory(),
                CCObj.liftedMulticutKernighanLinFactory(),
                CCObj.fusionMoveBasedFactory()
            ])

            fm = nlmc.PixelWiseLmcConnetedComponentsFusion2D(
                objective=self.cpp_obj(), 
                solver_factory=fm_factory)

            fused = fm.fuse(labels_a, labels_b)
    


            return fused
        else:
            raise NotImplementedError("only 3D atm")



