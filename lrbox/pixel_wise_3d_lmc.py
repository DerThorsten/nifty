import nifty
import nifty.graph.agglo
import nifty.segmentation
import numpy 
import h5py
import pylab
import nifty.graph.opt.lifted_multicut as nlmc


# mystuff
import data_loader
import make_weights
import objectives

import vigra
import nifty.segmentation as nseg


class Fuse3D(object):
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


        self.fm = nifty.graph.opt.lifted_multicut.PixelWiseLmcConnetedComponentsFusion3D(
            objective=self.objective.cpp_obj(), 
            solver_factory=solverFactory)

    def fuse_with(self, labels):

        labels = numpy.squeeze(labels)
        labels = numpy.require(labels, requirements=['C'])

        if labels.ndim == 3:
            if self.best_l is  None:
                self.best_l = labels
            else:
                #print("fuse 2")
                self.best_l = self.fm.fuse(
                    labels,
                    numpy.require(self.best_l,requirements=['C'])
                )
            
        else:
            labels = numpy.concatenate([self.best_l[...,None], labels],axis=3)
            self.best_l = self.fm.fuse(labels)

        self.best_e = self.objective.evaluate(self.best_l)
        print(self.best_e)






def save_seg(seg, raw, base_path):
    print("raw", raw.shape)
    print("seg", seg.shape)
    img = nseg.segmentOverlay(raw, seg, beta=0.3)
    #img = nseg.markBoundaries(raw, seg)
    vigra.impex.writeImage(img, base_path+"_overlay.png")


if __name__ == "__main__":


    # load the data
    mode = "test"
    slicing = [slice(0,30),slice(0,512), slice(0,512)]
    affinities, offsets, raw = data_loader.load_isbi_3d(slicing=slicing, mode=mode)
    shape = raw.shape
    # load all precomputed proposals
    precomputed_proposals = data_loader.load_predcomuted(slicing=slicing, mode=mode)


    # make lmc objective
    isbi_obj = objectives.IsbiObjective(offsets=offsets, affinities=affinities, raw=raw)



    GridObj = nlmc.LiftedMulticutObjectiveUndirectedGridGraph2DSimpleNh

    factory = GridObj.chainedSolversFactory([
        GridObj.liftedMulticutGreedyAdditiveFactory(),
        GridObj.liftedMulticutKernighanLinFactory()
        #GridObj.fusionMoveBasedFactory()
    ])

    isbi_obj_0 = isbi_obj.z_objective(z=29)



    def callback(labels_a, labels_b):
        save_seg(seg=labels_a, raw=isbi_obj_0.raw, base_path="/home/tbeier/src/nifty/lrbox/labels_a")
        save_seg(seg=labels_b, raw=isbi_obj_0.raw, base_path="/home/tbeier/src/nifty/lrbox/labels_b")

    seg = isbi_obj_0.optimize_blockwise(factory, callback=callback)
    save_seg(seg=seg, raw=isbi_obj_0.raw, base_path="/home/tbeier/src/nifty/lrbox/seg")








    pylab.imshow(nifty.segmentation.segmentOverlay(isbi_obj_0.raw, seg, showBoundaries=True))
    pylab.show()

    sys.exit(0)





    fuse_inf = Fuse3D(objective=isbi_obj, best_l=None)    

    while(True):
        print("OUTER")
        if fuse_inf.best_e is None:
            best_e_outer = float('inf')
        else:
            best_e_outer = float(fuse_inf.best_e)
        print("fuse with best") # fuse with best
        for x in range(2):
            #print("x",x)
            for p in precomputed_proposals:
                fuse_inf.fuse_with(p)
                #print("eval",isbi_obj.evaluate(p), fuse_inf.best_e)



        print("sub stuff")
        for p in precomputed_proposals:

            for z in range(shape[0]):

                proposal = fuse_inf.best_l.copy()
                max_l = proposal.max()
                proposal[z,...] = p[z,...] + max_l + 1

                fuse_inf.fuse_with(proposal)



        for z in range(shape[0]):

            

            for ps in (1,2):
                print("multi shiftey", ps)
                # shift
                while(True):

                    
                    proposal = fuse_inf.best_l.copy()
                    proposal_z = proposal[z,...]
                    mx = proposal_z.max()

                    best_e = float(fuse_inf.best_e)
                    padded = numpy.pad(proposal_z+1, ps+1, mode='constant', constant_values=0)

                    for x in range(-ps,ps+1):
                        for y in range(-ps,ps+1):


                            labels = padded[
                                ps + x :  ps + x + shape[1],
                                ps + y :  ps + y + shape[2],
                                
                            ]
                            proposal[z,...] = labels + mx
                            fuse_inf.fuse_with(proposal)
                
                    if(fuse_inf.best_e >= best_e):
                        break

        if(fuse_inf.best_e >= best_e_outer):
                 break


    # sanity check
    if False:
        pylab.imshow(aff[0,0,:,:])
        pylab.show()

        pylab.imshow(raw[0,:,:])
        pylab.show()


    best_l = fuse_inf.best_l

    res = h5py.File("/home/tbeier/nice_probs/plmc_tentative_%s.h5"%mode,'w')
    res['data'] = best_l
    res.close()



    dsws = precomputed_proposals[0]

    for z in (14,10,20,29):
        print(z)
        raw_z = raw[z,:,:]
        seg_lmc = nifty.segmentation.segmentOverlay(raw_z, best_l[z,:,:])
        seg_dws = nifty.segmentation.segmentOverlay(raw_z, dsws[z,:,:])

        # pylab.imshow(seg_dws)
        # pylab.show()

        fig = pylab.figure()

        ax = fig.add_subplot(1,3,1)
        ax.imshow(seg_lmc)

        ax = fig.add_subplot(1,3,2)
        ax.imshow(seg_dws)

        ax = fig.add_subplot(1,3,3)
        ax.imshow(raw_z)

  
    pylab.show()
