import nifty
import numpy
import nifty.segmentation
import vigra

numpy.random.seed(32)


Objective = nifty.graph.opt.lifted_multicut.PixelWiseLmcObjective2D


def solve_fm(objective, best_l=None):


    G = nifty.graph.UndirectedGraph
    CCObj = G.LiftedMulticutObjective 

    solverFactory = CCObj.liftedMulticutGreedyAdditiveFactory()
    fm = nifty.graph.opt.lifted_multicut.PixelWiseLmcConnetedComponentsFusion2D(objective=objective, solver_factory=solverFactory)




    best_e = None
    if best_l is not None:
        best_e = objective.evaluate(best_l)




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




def downsample_by_two(weights, offsets):
    n_offsets = offsets.shape[0]
    new_offsets = offsets.astype('float')/2.0
    new_dict = dict()

    def f(o):
        if(o>0.0 and o<1.0):
            return 1
        elif(o<0.0 and o>-1.0):
            return -1
        else:
            return int(round(o))



    for i_offset in range(n_offsets):
        weights_channel = weights[:,:,i_offset]
        shape = weights_channel.shape
        new_shape = [s//2 for s in shape]
        new_weights_channel = vigra.sampling.resize(weights_channel.astype('float32'), new_shape)

        offset = offsets[i_offset,:]
        nx,ny = new_offsets[i_offset,:]
        nx,ny = f(nx), f(ny)

        if (nx,ny) in new_dict:
            new_dict[(nx,ny)] += new_weights_channel
        else:
            new_dict[(nx,ny)] = new_weights_channel

        print(offset,(nx,ny))


    new_offsets = [ ]
    new_weights = [ ]

    for key in new_dict.keys():
        new_offsets.append(key)
        new_weights.append(new_dict[key])

    new_weights = numpy.array(new_weights)
    new_offsets = numpy.array(new_offsets)

    return numpy.rollaxis(new_weights,0,3),  new_offsets#numpy.swapaxes(new_offsets,0,1)



offsets = numpy.array([
    # local edges
    [ 0,  1],
    [ 0, -1],
    [ 1, -0],
    [-1, -0],
    # lifted
    [ 0,  2],
    [ 0, -2],
    [ 2, -0],
    [-2, -0],
    [ 0,  3],
    [ 0, -3],
    [ 3, -0],
    [-3, -0],
])

n_offsets = offsets.shape[0]
shape = [256, 256]
weights = numpy.random.random(size=shape + [n_offsets]) - 0.5001
print(offsets.shape,weights.shape)

weights2, offsets2 = downsample_by_two(weights=weights, offsets=offsets)

print(offsets2.shape,weights2.shape)
xsmall = solve_fm(Objective(weights2, offsets2))
print("best l ", xsmall.shape)

xsmall = vigra.taggedView(xsmall[:,:], 'xy')
xbig = vigra.sampling.resize(xsmall.astype('float32'), shape,0).astype('int')
xbig = numpy.array(xbig)
xbig = nifty.segmentation.connectedComponents(xbig)

#print("bruuu")
obj = Objective(weights, offsets)

print("from xbig",obj.evaluate(xbig))

xbig = solve_fm(obj, xbig)