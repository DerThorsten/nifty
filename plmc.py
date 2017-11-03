import nifty
import numpy
import nifty.segmentation
import vigra
import matplotlib.pyplot as plt

numpy.random.seed(32)




Objective = nifty.graph.opt.lifted_multicut.PixelWiseLmcObjective2D


class PlmcObjective2D(nifty.graph.opt.lifted_multicut.PixelWiseLmcObjective2D):
    def __init__(self, affinities, weights, offsets):
        self.affinities = affinities
        self.weights = weights
        self.offsets = offsets
        super(PlmcObjective2D, self).__init__(weights, offsets)



    def downsample_by_two(self):

        def impl(weights, affinities, offsets):
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

                shape = weights_channel.shape
                new_shape = [s//2 for s in shape]
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

            return numpy.rollaxis(new_weights,0,3), numpy.rollaxis(new_affinities,0,3),  new_offsets#numpy.swapaxes(new_offsets,0,1)


        new_weights,new_affinities, new_offsets = impl(weights=self.weights,affinities=self.affinities, offsets=self.offsets)

        return PlmcObjective2D(affinities=new_affinities, weights=new_weights, offsets=new_offsets)


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




def solve_single_scale(objective, best_l=None):


    class Fuse(object):
        def __init__(self,objective, best_l=None):
            self.objective = objective
            self.best_l = best_l
            self.best_e = None
            if self.best_l is not None:
                self.best_e = objective.evaluate(best_l)

            G = nifty.graph.UndirectedGraph
            CCObj = G.LiftedMulticutObjective 
            solverFactory = CCObj.liftedMulticutGreedyAdditiveFactory()
            self.fm = nifty.graph.opt.lifted_multicut.PixelWiseLmcConnetedComponentsFusion2D(
                objective=self.objective, 
                solver_factory=solverFactory)

        def fuse_with(self, labels):

            if self.best_l is  None:
                self.best_l = labels
            else:
                self.best_l = self.fm.fuse(labels, self.best_l)
            self.best_e = objective.evaluate(self.best_l)
            print(self.best_e)


    fuse_inf = Fuse(objective=objective, best_l=best_l)    

    local = local_affinities_to_pixel(objective.affinities, objective.offsets)

    def seeded_watersheds(sigma):
        hmap1 = vigra.filters.gaussianSmoothing(local, 0.2)
        hmap2 = vigra.filters.gaussianSmoothing(local, sigma)
        hmap1 += 0.03*hmap2
        seg = nifty.segmentation.seededWatersheds(hmap1, method='edge_weighted', acc='interpixel')
        return seg


    for x in range(2):


        for x in range(20):

            # random label
            labels = numpy.random.randint(0,3,size=objective.shape)
            labels = nifty.segmentation.connectedComponents(labels)
            fuse_inf.fuse_with(labels)


        fuse_inf.fuse_with(seeded_watersheds(0.1))
        fuse_inf.fuse_with(seeded_watersheds(0.2))
        fuse_inf.fuse_with(seeded_watersheds(0.3))
        fuse_inf.fuse_with(seeded_watersheds(0.5))
        fuse_inf.fuse_with(seeded_watersheds(1.0))
        fuse_inf.fuse_with(seeded_watersheds(2.5))
        fuse_inf.fuse_with(seeded_watersheds(3.5))
        fuse_inf.fuse_with(seeded_watersheds(4.5))
        fuse_inf.fuse_with(seeded_watersheds(5.5))






  



   
    plt.imshow(nifty.segmentation.markBoundaries(local, fuse_inf.best_l, color=(1,0,0)))
    plt.show()

    return best_l





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
    while(current.shape[0]!=64):
        print("jay")
        current = current.downsample_by_two()
        pyramid.append(current)
    pyramid = reversed(pyramid)

    old_res = None
    for obj in pyramid: 
        init = None
        if old_res is not None:
            print(old_res.shape)
            print('\n\n\n\n')
            init = vigra.sampling.resize(old_res.astype('float32'),0).astype('int')
        old_res = solve_single_scale(obj, init)

      

    sys.exit()




    for x in range(6):

        # random label
        labels = numpy.random.randint(0,2,size=shape)
        labels = nifty.segmentation.connectedComponents(labels)

        if best_l is None:
            best_l = labels
            best_e = objective.evaluate(labels)
            continue


        fused_labels = fm.fuse(labels, best_l)

        print("proposal ",objective.evaluate(labels))
        print("best     ",objective.evaluate(best_l))
        print("fused    ",objective.evaluate(fused_labels),"\n")

        e = objective.evaluate(fused_labels)
        
        if e < best_e :
            best_e = float(e)
            best_l = fused_labels.copy()
            #   print("best_e",best_e)

        #print("e=",objective.evaluate(labels))
    return best_l


def affinities_to_weights(affinities, offsets, beta=0.5):
    eps = 0.00001
    affinities = numpy.clip(affinities, eps, 1.0-eps)
    weights = numpy.log((1.0-affinities)/(affinities)) + numpy.log((1.0-beta)/(beta))
    return weights



  




def affinities_lmc(affinities, offsets, beta=0.5):

    # convert affinities to weights
    weights = affinities_to_weights(affinities=affinities, offsets=offsets, beta=0.5)

    w = numpy.sum(offsets**2,axis=1)
    weights *= w
    weights[:,:,0] =  0
    weights[:,:,1] =  0

    objective = PlmcObjective2D(affinities=affinities, weights=weights, offsets=offsets)

    solve_pyramid(objective)








if __name__ == "__main__":

    # load weighs and raw
    path_affinities = "/home/tbeier/nice_p/isbi_test_default.h5"
    offsets = numpy.array([
        [-1,0],[0,-1],
        [-9,0],[0,-9],[-9,-9],[9,-9],
        [-9,-4],[-4,-9],[4,-9],[9,-4],
        [-27,0],[0,-27],[-27,-27],[27,-27]
    ])
    import h5py
    f5_affinities = h5py.File(path_affinities)
    affinities = f5_affinities['data']
    z = 5
    # get once slice
    affinities = numpy.rollaxis(affinities[:,z,:,:],0,3)
    affinities = numpy.require(affinities, requirements=['C'])

    import skimage.io
    raw = skimage.io.imread('/home/tbeier/src/nifty/mysandbox/NaturePaperDataUpl/ISBI2012/raw_test.tif')
    raw = raw[z,:,:]


    raw = raw[0:256, 0:256]
    affinities = affinities[0:256, 0:256,:]

    print(raw.shape, affinities.shape)


    if False:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax1 = fig.add_subplot(2,1,1)
        ax1.imshow(raw)

        ax2 = fig.add_subplot(2,1,2)
        ax2.imshow(affinities[:,:,0])

        plt.show()

        sys.exit()


    affinities_lmc(affinities=affinities, offsets=offsets, beta=0.5)





