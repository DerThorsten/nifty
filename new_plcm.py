import nifty
import numpy
import nifty.segmentation
import nifty.graph.rag
import nifty.graph.agglo
import vigra
import matplotlib.pyplot as plt
from random import shuffle

#import fastfilters

numpy.random.seed(32)




Objective = nifty.graph.opt.lifted_multicut.PixelWiseLmcObjective2D


class PlmcObjective2D(nifty.graph.opt.lifted_multicut.PixelWiseLmcObjective2D):
    def __init__(self,raw, affinities, weights, offsets):
        self.raw = numpy.require(raw,dtype='float32')
        self.affinities = affinities
        self.weights = weights
        self.offsets = offsets
        super(PlmcObjective2D, self).__init__(weights, offsets)


    def proposals_from_raw(self):
        proposals = []

        for sigma in (1.0, 3.0, 5.0):

            raw = self.raw
            hmap = vigra.filters.hessianOfGaussianEigenvalues(raw, 5.0)[:,:,0]
            seg,nseg = vigra.analysis.watersheds(1.0*hmap)
            proposals.append(seg)


            #plt.imshow(nifty.segmentation.markBoundaries(raw/255.0, seg, color=(1,0,0)))
            #plt.show()

        return proposals


        

    def proposal_from_raw_agglo(self):

        proposals = []

      
        for sigma in (1.0, 3.0, 5.0):
            
            grow_map = vigra.filters.hessianOfGaussianEigenvalues(self.raw, sigma)[:,:,0]
            overseg,nseg = vigra.analysis.watersheds(grow_map.astype('float32'))


            rag = nifty.graph.rag.gridRag(overseg)
            edge_features, node_features = nifty.graph.rag.accumulateMeanAndLength(
                rag, grow_map, [512,512],0)
            meanEdgeStrength = edge_features[:,0]
            edgeSizes = edge_features[:,1]
            nodeSizes = node_features[:,1]


            for size_reg in (0.1,0.2,0.4,0.8):

                # cluster-policy  
                nnodes = rag.numberOfNodes//300
                nnodes = min(nnodes, 1000)
                clusterPolicy = nifty.graph.agglo.edgeWeightedClusterPolicy(
                    graph=rag, edgeIndicators=meanEdgeStrength,
                    edgeSizes=edgeSizes, nodeSizes=nodeSizes,
                    numberOfNodesStop=nnodes, sizeRegularizer=size_reg)

                # run agglomerative clustering
                agglomerativeClustering = nifty.graph.agglo.agglomerativeClustering(clusterPolicy) 
                agglomerativeClustering.run()
                nodeSeg = agglomerativeClustering.result()

                # convert graph segmentation
                # to pixel segmentation
                seg = nifty.graph.rag.projectScalarNodeDataToPixels(rag, nodeSeg)

                        
                #plt.imshow(nifty.segmentation.segmentOverlay(self.raw, seg, showBoundaries=False))
                #plt.show()

                proposals.append(seg)

        return proposals

    def proposal_from_local_agglo(self, hmap):

        proposals = []

        hmap0 = vigra.filters.gaussianSmoothing(hmap, 0.1)

        for sigma in (1.0, 3.0, 5.0):
            
            hmap1 = vigra.filters.gaussianSmoothing(hmap, sigma)
            grow_map = hmap0 + 0.05*hmap1
            overseg,nseg = vigra.analysis.watersheds(grow_map.astype('float32'))


            rag = nifty.graph.rag.gridRag(overseg)
            edge_features, node_features = nifty.graph.rag.accumulateMeanAndLength(
                rag, hmap, [512,512],0)
            meanEdgeStrength = edge_features[:,0]
            edgeSizes = edge_features[:,1]
            nodeSizes = node_features[:,1]


            for size_reg in (0.1,0.2,0.4,0.8):

                # cluster-policy  
                clusterPolicy = nifty.graph.agglo.edgeWeightedClusterPolicy(
                    graph=rag, edgeIndicators=meanEdgeStrength,
                    edgeSizes=edgeSizes, nodeSizes=nodeSizes,
                    numberOfNodesStop=rag.numberOfNodes//10, sizeRegularizer=size_reg)

                # run agglomerative clustering
                agglomerativeClustering = nifty.graph.agglo.agglomerativeClustering(clusterPolicy) 
                agglomerativeClustering.run()
                nodeSeg = agglomerativeClustering.result()

                # convert graph segmentation
                # to pixel segmentation
                seg = nifty.graph.rag.projectScalarNodeDataToPixels(rag, nodeSeg)

                        
                #plt.imshow(nifty.segmentation.segmentOverlay(self.raw, seg, showBoundaries=False))
                #plt.show()

                proposals.append(seg)

        return proposals





    def downsample_by_two(self):

        def impl(raw, weights, affinities, offsets):

            shape = weights.shape[0:2]
            new_shape = [s//2 for s in shape]

            new_raw = vigra.sampling.resize(raw.astype('float32'), new_shape)

            n_offsets = offsets.shape[0]
            new_offsets = offsets.astype('float')/2.0
            new_weight_dict = dict()
            new_affinity_dict = dict()
            def f(o):
                if(o>0.0 and o<1.0):
                    return 1
                elif(o<0.0 and o>-1.0):
                    return -1
                else:
                    return int(round(o))



            for i_offset in range(n_offsets):
                weights_channel = weights[:,:,i_offset]
                affinity_channel = affinities[:,:,i_offset]

                new_weights_channel = vigra.sampling.resize(weights_channel.astype('float32'), new_shape)
                new_affinity_channel = vigra.sampling.resize(affinity_channel.astype('float32'), new_shape)

                offset = offsets[i_offset,:]
                nx,ny = new_offsets[i_offset,:]
                nx,ny = f(nx), f(ny)

                if (nx,ny) in new_weight_dict:
                    new_weight_dict[(nx,ny)] += new_weights_channel
                    new_affinity_dict[(nx,ny)] += new_affinity_channel
                else:
                    new_weight_dict[(nx,ny)] = new_weights_channel
                    new_affinity_dict[(nx,ny)] = new_affinity_channel
                print(offset,(nx,ny))


            new_offsets = [ ]
            new_weights = [ ]
            new_affinities = [ ]
            for key in new_weight_dict.keys():
                new_offsets.append(key)
                new_weights.append(new_weight_dict[key])
                new_affinities.append(new_affinity_dict[key])
            new_weights = numpy.array(new_weights)
            new_affinities = numpy.array(new_affinities)
            new_offsets = numpy.array(new_offsets)

            return new_raw, numpy.rollaxis(new_weights,0,3), numpy.rollaxis(new_affinities,0,3),  new_offsets#numpy.swapaxes(new_offsets,0,1)


        new_raw, new_weights,new_affinities, new_offsets = impl(raw=self.raw,weights=self.weights,
            affinities=self.affinities, offsets=self.offsets)

        return PlmcObjective2D(raw=new_raw, affinities=new_affinities, weights=new_weights, offsets=new_offsets)





def local_affinities_to_pixel(affinities, offsets):
    
    shape = affinities.shape[0:2]

    offset_dict = dict()
    for i in range(offsets.shape[0]):
        x,y = offsets[i,:]
        key = int(x),int(y)
        offset_dict[key] = i


    local_edges = [
        (-1, 0),
        ( 1, 0),
        ( 0,-1),
        ( 0, 1)
    ]

    acc = numpy.zeros(shape)

    for local_edge in local_edges:
        #print("find",local_edge)

        if local_edge in offset_dict:

            acc += affinities[:,:, offset_dict[local_edge]]
        else:
            
            o_local_edge = tuple([-1*e for e in local_edge]) 
            #print("missing",local_edge)
            if o_local_edge in offset_dict:
                #print("    using: ",o_local_edge)
                o_channel = affinities[:,:, offset_dict[o_local_edge]]
                padded_o_channel = numpy.pad(o_channel, 1, mode='reflect')

                if local_edge == (0,1):
                    acc += padded_o_channel[1:shape[0]+1, 2:shape[1]+2]
                elif local_edge == (1,0):
                    acc += padded_o_channel[2:shape[0]+2, 1:shape[1]+1]
                elif local_edge == (0,-1):
                    acc += padded_o_channel[1:shape[0]+1, 0:shape[1]]
                elif local_edge == (1,0):
                    acc += padded_o_channel[0:shape[0], 1:shape[1]+1]
                else:
                    raise RuntimeError("todo")

    return acc

def make_pixel_wise(affinities, offsets):
    shape = affinities.shape[0:2]
    big_shape = tuple([2*s for s in shape])
    padding_size = int(numpy.abs(offsets).max())*2
    acc = numpy.zeros(shape)

    for i in range(offsets.shape[0]):
        print(i)
        affinity_channel = affinities[:, :, i]
        affinity_channel = vigra.sampling.resize(affinity_channel, big_shape)
        padded_affinity_channel = numpy.pad(affinity_channel, padding_size, mode='reflect')



        sx = padding_size - offsets[i,0]
        sy = padding_size - offsets[i,1]
        p_affinity = padded_affinity_channel[sx: sx+big_shape[0], sy: sy+big_shape[0]]
        sigma = 0.3*numpy.sum(offsets[i,:]**2)**0.5
        print("sigma",sigma)
        p_affinity = vigra.filters.gaussianSmoothing(p_affinity, sigma)

        acc += numpy.array(vigra.sampling.resize(p_affinity, shape))



    return acc

def solve_single_scale(objective, best_l=None):

    shape = objective.shape
    class Fuse(object):
        def __init__(self,objective, best_l=None):
            self.objective = objective
            self.best_l = best_l
            self.best_e = None
            if self.best_l is not None:
                self.best_e = objective.evaluate(best_l)

            G = nifty.graph.UndirectedGraph
            CCObj = G.LiftedMulticutObjective 




            greedySolverFactory = CCObj.liftedMulticutGreedyAdditiveFactory()
            klSolverFactory = CCObj.liftedMulticutKernighanLinFactory()

            solverFactory = CCObj.chainedSolversFactory([greedySolverFactory, greedySolverFactory])


            self.fm = nifty.graph.opt.lifted_multicut.PixelWiseLmcConnetedComponentsFusion2D(
                objective=self.objective, 
                solver_factory=solverFactory)

        def fuse_with(self, labels):

            labels = numpy.squeeze(labels)
            labels = numpy.require(labels, requirements=['C'])

            if labels.ndim == 2:
                if self.best_l is  None:
                    self.best_l = labels
                else:
                    #print("fuuuuu")
                    self.best_l = self.fm.fuse(
                        labels,
                        numpy.require(self.best_l,requirements=['C'])
                    )
                
            else:
                labels = numpy.concatenate([self.best_l[:,:,None], labels],axis=2)
                self.best_l = self.fm.fuse(labels)

            self.best_e = objective.evaluate(self.best_l)
            print(self.best_e)


    fuse_inf = Fuse(objective=objective, best_l=best_l)    

    local = local_affinities_to_pixel(objective.affinities, objective.offsets)

    def seeded_watersheds(sigma):
        #print("thesigma",sigma)
        hmap1 = vigra.filters.gaussianSmoothing(local, 0.2)
        hmap2 = vigra.filters.gaussianSmoothing(local, sigma)
        hmap1 += 0.03*hmap2
        #print(nifty.segmentation.seededWatersheds)
        seg = nifty.segmentation.seededWatersheds(hmap1, method='edge_weighted', acc='interpixel')

        return seg



    def refine_watershed(labels,r, sigma):


        hmap1 = vigra.filters.gaussianSmoothing(local, 0.2)
        hmap2 = vigra.filters.gaussianSmoothing(local, sigma)
        hmap1 += 0.03*hmap2


        zeros = numpy.zeros_like(labels)
        boundaries = skimage.segmentation.mark_boundaries(zeros, labels.astype('uint32'))[:,:,0]*255
        #print(boundaries.min(),boundaries.max())
        boundaries = vigra.filters.discDilation(boundaries.astype('uint8'),r).squeeze()
        new_seeds = labels + 1
        where_b = numpy.where(boundaries==1)
        new_seeds[boundaries==255] = 0
        seg,nseg = vigra.analysis.watersheds(hmap1.astype('float32'), seeds=new_seeds.astype('uint32'))
        seg = nifty.segmentation.connectedComponents(seg)
        return seg


    def refiner(labels,r):

        grid = numpy.arange(labels.size) + labels.max() + 1
        grid = grid.reshape(labels.shape)

        zeros = numpy.zeros_like(labels)
        boundaries = skimage.segmentation.mark_boundaries(zeros, labels.astype('uint32'))[:,:,0]*255


        #print(boundaries.min(),boundaries.max())
        boundaries = vigra.filters.discDilation(boundaries.astype('uint8'),r).squeeze()
        new_seeds = labels.copy()
        where_mask = boundaries==255
        new_seeds[where_mask] = grid[where_mask]
        
        return new_seeds











    proposals = []

    proposals += objective.proposals_from_raw()
    proposals += objective.proposal_from_local_agglo(local)
    proposals += objective.proposal_from_raw_agglo()
    proposals += [seeded_watersheds(s) for s in (1.0, 2.0, 3.0)]


    #shuffle(proposals)


    print("fuabsf")




    for proposal in proposals:
        print("fuse with prop")
        fuse_inf.fuse_with(proposal)

    
    
    if False:
        print("refine watershed")

        for r in (1,2,3,4,5):
            for s in (1.0, 2.0, 3.0,5.0):
                p = refine_watershed(fuse_inf.best_l,r=r,sigma=s)
                fuse_inf.fuse_with(p)

    else:
        for r in (1,2,3,4):
            while(True):
                print("buja",r)
                best_e = float(fuse_inf.best_e)
                fuse_inf.fuse_with(refiner(fuse_inf.best_l, r=2))
                if fuse_inf.best_e >= best_e:
                    break

    #sys.exit()


    if True:

        for ps in (1,2,3,4):
            print("multi shiftey", ps)
            # shift
            for i in range(10):


                print("Si",i)

                proposals = []


                best_e = float(fuse_inf.best_e)
                padded = numpy.pad(fuse_inf.best_l+1, ps+1, mode='constant', constant_values=0)
                for x in range(-ps,ps+1):
                    for y in range(-ps,ps+1):


                        labels = padded[
                            ps + x :  ps + x + shape[0],
                            ps + y :  ps + y + shape[1]
                        ]
                        #labels = nifty.segmentation.connectedComponents(prop)
                        proposals.append(labels[:,:,None])

                        if len(proposals) >= 6:
                            proposals = numpy.concatenate(proposals, axis=2)
                            fuse_inf.fuse_with(proposals)
                            proposals = []

                
                if len(proposals) >= 1:

                    proposals = numpy.concatenate(proposals, axis=2)
                    fuse_inf.fuse_with(proposals)

                if(fuse_inf.best_e >= best_e):
                    break


        print("shiftey done ")

    else:



        print("shiftey")
        # shift
        ps = 2
        for i in range(10):
            print("Si",i)

            proposals = []


            best_e = float(fuse_inf.best_e)
            padded = numpy.pad(fuse_inf.best_l+1, ps+1, mode='constant', constant_values=0)
            for x in range(-ps,ps):
                for y in range(-ps,ps):

                    labels = padded[
                        ps + x :  ps + x + shape[0],
                        ps + y :  ps + y + shape[1]
                    ]
                    #labels = nifty.segmentation.connectedComponents(prop)
                    proposals.append(labels)
            
            shuffle(proposals)
            for labels in proposals:
                fuse_inf.fuse_with(labels)

            if(fuse_inf.best_e >= best_e):
                break

        print("shiftey done ")




  



    return fuse_inf.best_l





def solve_pyramid(objective, best_l=None):

    



    G = nifty.graph.UndirectedGraph
    CCObj = G.LiftedMulticutObjective 

    solverFactory = CCObj.liftedMulticutGreedyAdditiveFactory()
    fm = nifty.graph.opt.lifted_multicut.PixelWiseLmcConnetedComponentsFusion2D(objective=objective, solver_factory=solverFactory)

    shape = objective.shape


    best_e = None
    if best_l is not None:
        best_e = objective.evaluate(best_l)




    # make a pyramid
    current = objective
    pyramid = [current]
    #while(current.shape[0]!=64):
    #    print("jay")
    #    current = current.downsample_by_two()
    #    pyramid.append(current)
    #pyramid = reversed(pyramid)





    old_res = None
    for obj in pyramid: 
        init = None
        if old_res is not None:
            print(old_res.shape)
            print('\n\n\n\n')
            init = vigra.sampling.resize(old_res.astype('float32'), obj.shape ,0).astype('int')
        old_res = solve_single_scale(obj, init)

    res = old_res

    return res


def affinities_to_weights(affinities, offsets, beta=0.5):
    eps = 0.00001
    affinities = numpy.clip(affinities, eps, 1.0-eps)
    weights = numpy.log((1.0-affinities)/(affinities)) + numpy.log((1.0-beta)/(beta))
    return weights



  
def affinities_to_better_weights(affinities, offsets, beta=0.5):
    weights = affinities.copy()

    eps = 0.00001
    affinities = numpy.clip(affinities, eps, 1.0-eps)
    weights = numpy.log((1.0-affinities)/(affinities)) + numpy.log((1.0-beta)/(beta))


    # long range
    weights[:,:,:]  = -1.0*(affinities[:,:,:]-0.5)

    # local weighs
    weights[:,:,0] = 1.0 - affinities[:,:,0]
    weights[:,:,1] = 1.0 - affinities[:,:,1]

    weights *= numpy.sum(offsets**2,1)**0.5


    return weights




def affinities_lmc(raw, affinities, offsets, beta=0.5):

    # convert affinities to weights
    weights = affinities_to_better_weights(affinities=affinities, offsets=offsets, beta=0.5)

    #w = numpy.sum(offsets**2,axis=1)
    #weights *= w
    #weights[:,:,0] =  0
    #weights[:,:,1] =  0

    objective = PlmcObjective2D(raw=raw, affinities=affinities, weights=weights, offsets=offsets)

    return solve_pyramid(objective)







if __name__ == "__main__":

    # load weighs and raw
    path_affinities = "/home/tbeier/nice_p/isbi_test_default.h5"
    #path_affinities = "/home/tbeier/nice_probs/isbi_test_default.h5"
    offsets = numpy.array([
        [-1,0],[0,-1],
        [-9,0],[0,-9],[-9,-9],[9,-9],
        [-9,-4],[-4,-9],[4,-9],[9,-4],
        [-27,0],[0,-27],[-27,-27],[27,-27]
    ])
    import h5py
    f5_affinities = h5py.File(path_affinities)
    affinities = f5_affinities['data']
    z = 8
    # get once slice
    affinities = numpy.rollaxis(affinities[:,z,:,:],0,3)
    affinities = numpy.require(affinities, requirements=['C'])

    import skimage.io
    #raw_path = "/home/tbeier/src/nifty/src/python/examples/multicut/NaturePaperDataUpl/ISBI2012/raw_test.tif"
    raw_path = '/home/tbeier/src/nifty/mysandbox/NaturePaperDataUpl/ISBI2012/raw_test.tif'
    raw = skimage.io.imread(raw_path)
    raw = raw[z,:,:]


    #raw = raw[200:64+200, 200:64+200]
    #affinities = affinities[200:64+200, 200:64+200,:]

    #t = 0.2
    #affinities[affinities >= t ] = 1 
    #affinities[affinities <  t ] = 0 

    print(raw.shape, affinities.shape)





    if False:
        import matplotlib.pyplot as plt

        for x in range(offsets.shape[0]):
            fig = plt.figure()
            ax1 = fig.add_subplot(2,1,1)
            ax1.imshow(raw)

            ax2 = fig.add_subplot(2,1,2)
            ax2.imshow(affinities[:,:,x])

            plt.show()

        sys.exit()


    res = affinities_lmc(raw=raw, affinities=affinities, offsets=offsets, beta=0.5)



    plt.imshow(nifty.segmentation.segmentOverlay(raw, res, showBoundaries=False))
    plt.show()

  
    plt.imshow(nifty.segmentation.markBoundaries(raw, res, color=(1,0,0)))
    plt.show()


