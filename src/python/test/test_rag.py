from __future__ import print_function
import nifty
import numpy
import nifty

def testInsert():

    labels = numpy.zeros(shape=[2,2],dtype='uint32')

    labels[0,0] = 0 
    labels[1,0] = 1 
    labels[0,1] = 0 
    labels[1,1] = 2 

    g =  nifty.graph.rag.explicitLabelsGridRag2D(labels)
    weights = numpy.ones(g.numberOfEdges)*1
    obj = nifty.graph.multicut.multicutObjective(g, weights)


    greedy=obj.greedyAdditiveFactory().create(obj)
    visitor = obj.multicutVerboseVisitor()
    ret = greedy.optimize()
    #print("greedy",obj.evalNodeLabels(ret))




    assert g.numberOfNodes == 3
    assert g.numberOfEdges == 3

    insertWorked = True
    try:
        g.insertEdge(0,1)
    except:
        insertWorked = False
    assert insertWorked == False


def testExplicitLabelsRag2d():

    labels = [
        [0,1,2],
        [0,0,2],
        [3,3,2],
        [4,4,4]
    ]

    ragA = nifty.graph.rag.gridRag(labels,0)
    ragB = nifty.graph.rag.gridRag(labels)

    assert isinstance(ragA, nifty.graph.rag.ExplicitLabelsGridRag2D)
    assert isinstance(ragB, nifty.graph.rag.ExplicitLabelsGridRag2D)

    shoudlEdges = [
        (0,1),
        (0,2),
        (0,3),
        (1,2),
        (2,3),
        (2,4),
        (3,4)
    ]

    shoudlNotEdges = [
        (0,4),
        (1,3),
        (1,4)
    ]


    assert ragA.numberOfNodes == 5
    assert ragB.numberOfNodes == 5

    assert ragA.numberOfEdges == len(shoudlEdges)
    assert ragB.numberOfEdges == len(shoudlEdges)


    edgeListA = []
    for edge in ragA.edges():
        edgeListA.append(edge)

    edgeListB = []
    for edge in ragB.edges():
        edgeListB.append(edge)

    assert len(edgeListA) == len(shoudlEdges)
    assert len(edgeListB) == len(shoudlEdges)  


    for shouldEdge in shoudlEdges:

        fResA = ragA.findEdge(shouldEdge)
        fResB = ragB.findEdge(shouldEdge)
        assert fResA >= 0
        assert fResB >= 0
        uvA = ragA.uv(fResA)
        uvB = ragB.uv(fResB)
        uvA = sorted(uvA)
        uvB = sorted(uvB)
        assert uvA[0] == shouldEdge[0]
        assert uvA[1] == shouldEdge[1]
        assert uvB[0] == shouldEdge[0]
        assert uvB[1] == shouldEdge[1]

    for shouldNotEdge in shoudlNotEdges:

        fResA = ragA.findEdge(shouldNotEdge)
        fResB = ragB.findEdge(shouldNotEdge)
        assert fResA == -1
        assert fResB == -1


def testExplicitLabelsRag2dSerializeDeserialize():

    labels = [
        [0,1,2],
        [0,0,2],
        [3,3,2],
        [4,4,4]
    ]

    ragA = nifty.graph.rag.gridRag(labels,0)
    ragB = nifty.graph.rag.gridRag(labels)

    assert isinstance(ragA, nifty.graph.rag.ExplicitLabelsGridRag2D)
    assert isinstance(ragB, nifty.graph.rag.ExplicitLabelsGridRag2D)

    shoudlEdges = [
        (0,1),
        (0,2),
        (0,3),
        (1,2),
        (2,3),
        (2,4),
        (3,4)
    ]

    shoudlNotEdges = [
        (0,4),
        (1,3),
        (1,4)
    ]


    assert ragA.numberOfNodes == 5
    assert ragB.numberOfNodes == 5

    assert ragA.numberOfEdges == len(shoudlEdges)
    assert ragB.numberOfEdges == len(shoudlEdges)


    edgeListA = []
    for edge in ragA.edges():
        edgeListA.append(edge)

    edgeListB = []
    for edge in ragB.edges():
        edgeListB.append(edge)

    assert len(edgeListA) == len(shoudlEdges)
    assert len(edgeListB) == len(shoudlEdges)  


    for shouldEdge in shoudlEdges:

        fResA = ragA.findEdge(shouldEdge)
        fResB = ragB.findEdge(shouldEdge)
        assert fResA >= 0
        assert fResB >= 0
        uvA = ragA.uv(fResA)
        uvB = ragB.uv(fResB)
        uvA = sorted(uvA)
        uvB = sorted(uvB)
        assert uvA[0] == shouldEdge[0]
        assert uvA[1] == shouldEdge[1]
        assert uvB[0] == shouldEdge[0]
        assert uvB[1] == shouldEdge[1]

    for shouldNotEdge in shoudlNotEdges:

        fResA = ragA.findEdge(shouldNotEdge)
        fResB = ragB.findEdge(shouldNotEdge)
        assert fResA == -1
        assert fResB == -1




def testExplicitLabelsRag3d():

    labels = [
        [
            [0,1],
            [0,0]
        ],
        [
            [1,1],
            [2,2]
        ],
        [
            [3,3],
            [3,3]
        ]
    ]

    labels = numpy.array(labels)

    ragA = nifty.graph.rag.gridRag(labels,0)
    ragB = nifty.graph.rag.gridRag(labels)

    assert isinstance(ragA, nifty.graph.rag.ExplicitLabelsGridRag3D)
    assert isinstance(ragB, nifty.graph.rag.ExplicitLabelsGridRag3D)


    shoudlEdges = [
        (0,1),
        (0,2),
        (1,2),
        (1,3),
        (2,3)
        
    ]

    shoudlNotEdges = [
       (0,3)
    ]


    assert ragA.numberOfNodes == 4
    assert ragB.numberOfNodes == 4

    assert ragA.numberOfEdges == len(shoudlEdges)
    assert ragB.numberOfEdges == len(shoudlEdges)


    edgeListA = []
    for edge in ragA.edges():
        edgeListA.append(edge)

    edgeListB = []
    for edge in ragB.edges():
        edgeListB.append(edge)

    assert len(edgeListA) == len(shoudlEdges)
    assert len(edgeListB) == len(shoudlEdges)  


    for shouldEdge in shoudlEdges:

        fResA = ragA.findEdge(shouldEdge)
        fResB = ragB.findEdge(shouldEdge)
        assert fResA >= 0
        assert fResB >= 0
        uvA = ragA.uv(fResA)
        uvB = ragB.uv(fResB)
        uvA = sorted(uvA)
        uvB = sorted(uvB)
        assert uvA[0] == shouldEdge[0]
        assert uvA[1] == shouldEdge[1]
        assert uvB[0] == shouldEdge[0]
        assert uvB[1] == shouldEdge[1]

    for shouldNotEdge in shoudlNotEdges:

        fResA = ragA.findEdge(shouldNotEdge)
        fResB = ragB.findEdge(shouldNotEdge)
        assert fResA == -1
        assert fResB == -1
