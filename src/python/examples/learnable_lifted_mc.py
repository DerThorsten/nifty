import nifty

import vigra
import numpy

import nifty.graph.lifted_multicut as nifty_lmc
import nifty.structured_learning as nifty_sl

def gridGraph(shape):

    def nid(x, y):
        return x*shape[1] + y

    G = nifty.graph.UndirectedGraph
    g =  G(shape[0] * shape[1])
    for x in range(shape[0]):
        for y in range(shape[1]):  

            u = nid(x,y)

            if x + 1 < shape[0]:

                v = nid(x+1, y)
                g.insertEdge(u, v)

            if y + 1 < shape[1]:

                v = nid(x, y+1)
                g.insertEdge(u, v)

    return g, nid

    
numpy.random.seed(42)

def gererateToyData(n, shape=[30,30], noise=3):
    rawImages = []
    gtImages = []
    for i in range(n):

        gtImg = numpy.zeros(shape)
        gtImg[0:shape[0]/2,:] = 1

        gtImg[shape[0]/4: 3*shape[0]/4, shape[0]/4: 3*shape[0]/4]  = 2

        ra = numpy.random.randint(180)
        #print ra 

        gtImg = vigra.sampling.rotateImageDegree(gtImg.astype(numpy.float32),int(ra),splineOrder=0)
        gtImg = vigra.analysis.labelImage(gtImg.astype('uint32'))
        if True and i==0 :
            vigra.imshow(gtImg)
            vigra.show()

        img = gtImg + numpy.random.random(shape)*float(noise)
        if True and i==0 :
            vigra.imshow(img)
            vigra.show()

        rawImages.append(img.astype('float32'))
        gtImages.append(gtImg)

    return rawImages, gtImages



shape = [40,40]

# classes





sigmas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
maxGraphDist = 3



numberOfWeights = 2 #nDistances * (len(sigmas)+1)  +1
numberOfWeights = len(sigmas)  + 1


# get the solver factory for the loss augmented model
WeightedObj = nifty.graph.UndirectedGraph.WeightedLiftedMulticutObjective
LossAugmentedObj = WeightedObj.LossAugmentedViewLiftedMulticutObjective
pgen = LossAugmentedObj.watershedProposalGenerator('SEED_FROM_LOCAL')


factoryA = LossAugmentedObj.liftedMulticutGreedyAdditiveFactory()
factoryB = LossAugmentedObj.fusionMoveBasedFactory(proposalGenerator=pgen)
solverFactory = LossAugmentedObj.chainedSolverFactory(factoryA, factoryB)
oracle = nifty_sl.StructMaxMarginOracleLmc(solverFactory=solverFactory,numberOfWeights=numberOfWeights)
structMaxMargin = nifty_sl.structMaxMargin(oracle)




def nodeToCoord(node):


    x = int(node // shape[1])
    y = int(node - x*shape[1])

    return x,y



def fillObj(obj, rawImage, numberOfWeights):



    imgs = [rawImage] 
    imgs = []
    for sigma in sigmas:
        f = numpy.require(vigra.filters.gaussianSmoothing(rawImage, sigma),requirements=['C_CONTIGUOUS'])   
        imgs.append(f)


    for uv,dist in zip(uvIds,distances):

        uv = int(uv[0]),int(uv[1])

        coordU = nodeToCoord(uv[0])
        coordV = nodeToCoord(uv[1])



        for imgIndex, img in enumerate(imgs):
            feat = abs(img[coordU] - img[coordV])
            distIndex = dist - 1
            #wIndex = int(imgIndex*(nDistances) + distIndex)
            #print "wIndex",wIndex
            #print uv
            obj.addWeightedFeature(uv[0], uv[1], imgIndex, feat)



        obj.addWeightedFeature(uv[0], uv[1], numberOfWeights-1, 1.0)

        # add const term (this is not weighted)
        obj.setConstTerm(uv[0], uv[1],  0.0)



# raw data and gt vectors
rawImages, gtImages = gererateToyData(10,shape=shape)



for i,(rawImage, gtImage) in enumerate(zip(rawImages, gtImages)):


    graph, nodeIndex = gridGraph(shape)
    uvIds, distances  = graph.bfsNodes(maxGraphDist)



    nodeGt = gtImage.reshape([-1])
    loss =distances**2
    oracle.addModel(graph=graph, edges=uvIds, nodeGroundTruth=nodeGt, loss=loss)

    obj = oracle.getWeightedModel(i)
    fillObj(obj,rawImage,numberOfWeights)





structMaxMargin.learn()
weights = structMaxMargin.getWeights()

print weights


def optimize(ob):
    solverFactory = obj.liftedMulticutKernighanLinFactory()
    solver = solverFactory.create(obj)
    visitor = obj.verboseVisitor()
    argN = solver.optimize()
    
    print "KL",obj.evalNodeLabels(argN)



    pgen = obj.watershedProposalGenerator('SEED_FROM_LOCAL')
    solverFactory = obj.fusionMoveBasedFactory(proposalGenerator=pgen)
    solver = solverFactory.create(obj)
    visitor = obj.verboseVisitor()
    argN = solver.optimize(argN.copy())
    print "FM",obj.evalNodeLabels(argN)
    return argN
    



# some test models
rawImages, gtImages = gererateToyData(10,shape=shape)


for i,(rawImage, gtImage) in enumerate(zip(rawImages, gtImages)):


    graph, nodeIndex = gridGraph(shape)
    uvIds, distances = graph.bfsNodes(maxGraphDist)



    nodeGt = gtImage.reshape([-1])

    obj = nifty_lmc.weightedLiftedMulticutObjective(graph, numberOfWeights, uvIds)

    fillObj(obj,rawImage,numberOfWeights)
    obj.changeWeights(weights)






    y = optimize(obj)
    y = y.reshape(shape)

    vigra.imshow(gtImage)
    vigra.show()


    vigra.imshow(y)
    vigra.show()



