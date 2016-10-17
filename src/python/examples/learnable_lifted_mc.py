import nifty

import vigra
import numpy

import nifty.graph.lifted_multicut as nifty_lmc

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

def gererateToyDataset(n, shape=[30,30], noise=3):
    rawImages = []
    gtImages = []
    for i in range(n):

        gtImg = numpy.zeros(shape)
        gtImg[0:shape[0]/2,:] = 1

        gtImg[shape[0]/4: 3*shape[0]/4, shape[0]/4: 3*shape[0]/4]  = 2

        ra = numpy.random.randint(180)
        #print ra 

        gtImg = vigra.sampling.rotateImageDegree(gtImg.astype(numpy.float32),int(ra),splineOrder=0)

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


# raw data and gt vectors
rawImages, gtImages = gererateToyDataset(2,shape=shape)


sigmas = [1.0, .5]
maxGraphDist = 4

# since distances start at 1
nDistances = maxGraphDist

nWeights = 2 #nDistances * (len(sigmas)+1)  +1
nWeights = len(sigmas)  + 1
def nodeToCoord(node):


    x = int(node // shape[1])
    y = int(node - x*shape[1])

    return x,y


dataset = []
weights = numpy.zeros(nWeights)
for rawImage, gtImages in zip(rawImages, gtImages):

    shape = rawImage.shape
    graph, nodeIndex = gridGraph(shape)

    uvIds, distances = graph.bfsNodes(maxGraphDist)


    #print distances.max(),distances.min()
    GraphCls = graph.__class__
    LiftedObjective = graph.WeightedLiftedMulticutObjective

    

    obj = nifty_lmc.weightedLiftedMulticutObjective(graph, nWeights)
    nodeGt = gtImages.reshape([-1])
    nodeSizes = numpy.ones(obj.graph.numberOfNodes)
    lossAugmentedObj = nifty_lmc.lossAugmentedViewLiftedMulticutObjective(obj, nodeGt, nodeSizes)
    

    dataset.append((obj,lossAugmentedObj,nodeGt))


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
            wIndex = int(imgIndex*(nDistances) + distIndex)
            #print "wIndex",wIndex
            #print uv
            obj.addWeightedFeature(uv[0], uv[1], imgIndex, feat)


        # add gt feature
        #isCut = nodeGt[uv[0]] != nodeGt[uv[1]]
        #isCut2 = gtImages[nodeToCoord(uv[0])] != gtImages[nodeToCoord(uv[1])]
        #assert isCut == isCut2
        #obj.addWeightedFeature(uv[0], uv[1], 0, float(isCut))

        # add const feature
        obj.addWeightedFeature(uv[0], uv[1], nWeights-1, 1.0)

        # add const term (this is not weighted)
        #obj.setConstTerm(uv[0], uv[1],  10000.0)


    # initialize weights
    
    lossAugmentedObj.changeWeights(weights)



    allCut = numpy.arange(obj.liftedGraph.numberOfNodes)







# tiny ssvm 
C = 1
lrate = 1

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
    
def modelLoss(obj, gt):
    res = optimize(obj)
    #print res
    g = obj.liftedGraph

    s = 0.0
    c = 0
    for edge in g.edges():
        uv = g.uv(edge)

        egt  = gt[uv[0]] != gt[uv[1]]
        esol = res[uv[0]] != res[uv[1]]

        if egt != esol:
            s += 1.0
        c +=1
    return s/float(c)

def datsetLoss(dataset):
    s = 0.0
    for obj,lossAugmentedObj, gt in dataset:
        s += modelLoss(obj, gt)

    return s/len(dataset)


def allfinite(x):
    return numpy.isfinite(x).sum() == len(x)


for x in range(200):
    t = float(x+1)
    sols = []

    # get sols
    for obj,lossAugmentedObj, gt in dataset:
        y = optimize(lossAugmentedObj)
        sols.append(y)

    # gradient
    gradient = numpy.zeros(nWeights)
    for (obj,lossAugmentedObj, gt),sol in zip(dataset,sols):

        fSol = obj.getGradient(sol)
        fGt  = obj.getGradient(gt)

        assert allfinite(fSol)
        assert allfinite(fGt)

        g =  fGt-fSol
        gradient += g 

    g = weights + (C/float(len(dataset)) )*gradient

    elrate = lrate/t
    weights  = weights - elrate*g

    #update weights
    for (obj,lossAugmentedObj, gt),sol in zip(dataset,sols):
        lossAugmentedObj.changeWeights(weights)

    print "LOSS",datsetLoss(dataset)
    print "W", weights

    if (x+1) % 10 == 0:

        for obj,lossAugmentedObj, gt in dataset:
            y = optimize(obj)
            y = y.reshape(shape)
            vigra.imshow(y)
            vigra.show()