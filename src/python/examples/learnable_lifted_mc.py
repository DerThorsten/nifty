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



shape = [20,20]

# classes


# raw data and gt vectors
rawImages, gtImages = gererateToyDataset(10,shape=shape)


learnabelModels = []
for rawImage in rawImages:

    shape = rawImage.shape
    graph, nodeIndex = gridGraph(shape)

    GraphCls = graph.__class__
    LiftedObjective = graph.WeightedLiftedMulticutObjective

    

    obj = nifty_lmc.weightedLiftedMulticutObjective(graph)
    
    