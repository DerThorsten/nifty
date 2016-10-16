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

        if False and i==0 :
            vigra.imshow(gtImg)
            vigra.show()

        img = gtImg + numpy.random.random(shape)*float(noise)
        if False and i==0 :
            vigra.imshow(img)
            vigra.show()

        rawImages.append(img.astype('float32'))
        gtImages.append(gtImg)

    return rawImages, gtImages



shape = [6,6]

# classes


# raw data and gt vectors
rawImages, gtImages = gererateToyDataset(1,shape=shape)


sigmas = []#[0.5 ,1.0, 1.5]
maxGraphDist = 3

# since distances start at 1
nDistances = maxGraphDist

nWeights = nDistances * (len(sigmas)+1)

def nodeToCoord(node):


    x = int(node // shape[1])
    y = int(node - x*shape[1])

    return x,y


learnabelModels = []
for rawImage, gtImages in zip(rawImages, gtImages):

    shape = rawImage.shape
    graph, nodeIndex = gridGraph(shape)

    uvIds, distances = graph.bfsNodes(maxGraphDist)


    #print distances.max(),distances.min()
    GraphCls = graph.__class__
    LiftedObjective = graph.WeightedLiftedMulticutObjective

    

    obj = nifty_lmc.weightedLiftedMulticutObjective(graph)
    nodeGt = gtImages.reshape([-1])
    nodeSizes = numpy.ones(obj.graph.numberOfNodes)
    lossAugmentedObj = nifty_lmc.lossAugmentedViewLiftedMulticutObjective(obj, nodeGt, nodeSizes)
        


    imgs = [rawImage] 
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
            obj.addWeightedFeature(uv[0], uv[1], wIndex, feat)





# initialize weights
weights = numpy.ones(nWeights+100)
lossAugmentedObj.changeWeights(weights)



allCut = numpy.arange(obj.liftedGraph.numberOfNodes)
print obj.evalNodeLabels(allCut)
print lossAugmentedObj.evalNodeLabels(allCut)

