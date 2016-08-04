from __future__ import print_function
import nifty
import numpy
import nifty
import tempfile
import shutil
import os 

nrag = nifty.graph.rag

def ensureDir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

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


if nifty.Configuration.WITH_HDF5:

    nhdf5 = nifty.hdf5

    def testHdf5Rag2d():
        tempFolder = tempfile.mkdtemp()
        ensureDir(tempFolder)
        fpath = os.path.join(tempFolder,'_nifty_test_array4_.h5')
        

        try:
            
            shape = [3,3]
            chunkShape = [1,1]
            blockShape =  [2,2]

            hidT = nhdf5.createFile(fpath)
            array = nhdf5.Hdf5ArrayUInt32(hidT, "data", shape, chunkShape)

            assert array.shape[0] == shape[0]
            assert array.shape[1] == shape[1]

            labels = numpy.array( [
                [0,  1, 1],
                [2,  2, 2],
                [3,  3, 3]
            ],dtype='uint32')

            assert labels.shape[0] == shape[0]
            assert labels.shape[1] == shape[1]

            print(labels.shape)

            array[0:shape[0], 0:shape[1]] = labels
            resLabels = array[0:shape[0],0:shape[1]]

            print (resLabels)

            assert resLabels.min() == labels.min()
            assert resLabels.max() == labels.max()
            assert numpy.array_equal(resLabels, labels) == True
            
            labelsProxy = nrag.gridRagHdf5LabelsProxy(array, int(labels.max()+1))
            rarray = labelsProxy.hdf5Array()
            resLabels2 = rarray[0:shape[0],0:shape[1]]
            print (resLabels2)




            print("compute rag")
            rag = nrag.gridRagHdf5(labelsProxy,blockShape,1)


            r3 = rag.labelsProxy().hdf5Array()
            resLabels3 = r3[0:shape[0],0:shape[1]]
            print (resLabels3)


            # test the rag itself
            print("test")
            shoudlEdges = [
                (0,1),
                (0,2),
                (1,2),
                (2,3)
            ]

            shoudlNotEdges = [
                (0,3),
                (1,3)
            ]

            print("rag node num",rag.numberOfNodes)
            print("rag edge num",rag.numberOfEdges) 
            assert rag.numberOfNodes == labels.max()+1
            assert rag.numberOfEdges == len(shoudlEdges)



            edgeList = []
            for edge in rag.edges():
                edgeList.append(edge)

            assert len(edgeList) == len(shoudlEdges)

            for shouldEdge in shoudlEdges:

                fRes = rag.findEdge(shouldEdge)
                assert fRes >= 0
                uv = rag.uv(fRes)
                uv = sorted(uv)
                assert uv[0] == shouldEdge[0]
                assert uv[1] == shouldEdge[1]

            for shouldNotEdge in shoudlNotEdges:
                fRes = rag.findEdge(shouldNotEdge)
                assert fRes == -1

            
       
        finally:
            try:
                os.remove(fpath)
                shutil.rmtree(tempFolder)

            except:
                pass












if False:#nifty.Configuration.WITH_HDF5:
    def testHdf5Rag2d():
        tempFolder = tempfile.mkdtemp()
        ensureDir(tempFolder)
        fpath = os.path.join(tempFolder,'_nifty_test_array4_.h5')
        

        try:
            
            shape = [2,2]
            hidT = nhdf5.createFile(fpath)
            array = nhdf5.Hdf5ArrayUInt32(hidT, "data", shape, [1,1])

            assert array.shape[0] == 2
            assert array.shape[1] == 2

            labels = numpy.array( [
                [0,  1,],
                [2,  3,]
            ],dtype='uint32')

            print(labels.shape)

            array[0:shape[0], 0:shape[1]] = labels
            resLabels = array[0:shape[0],0:shape[1]]

            print (resLabels)

            assert resLabels.min() == labels.min()
            assert resLabels.max() == labels.max()
            assert numpy.array_equal(resLabels, labels) == True
            
            labelsProxy = nrag.gridRagHdf5LabelsProxy(array, int(labels.max()+1))
            rarray = labelsProxy.hdf5Array()
            resLabels2 = rarray[0:shape[0],0:shape[1]]
            print (resLabels2)




            print("compute rag")
            rag = nrag.gridRagHdf5(labelsProxy,[2,2],1)


            r3 = rag.labelsProxy().hdf5Array()
            resLabels3 = r3[0:shape[0],0:shape[1]]
            print (resLabels3)


            # test the rag itself
            print("test")
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

            print("rag node num",rag.numberOfNodes)
            print("rag edge num",rag.numberOfEdges) 
            assert rag.numberOfNodes == labels.max()+1
            assert rag.numberOfEdges == len(shoudlEdges)



            edgeList = []
            for edge in rag.edges():
                edgeList.append(edge)

            assert len(edgeList) == len(shoudlEdges)

            for shouldEdge in shoudlEdges:

                fRes = rag.findEdge(shouldEdge)
                assert fRes >= 0
                uv = rag.uv(fRes)
                uv = sorted(uv)
                assert uv[0] == shouldEdge[0]
                assert uv[1] == shouldEdge[1]

            for shouldNotEdge in shoudlNotEdges:
                fRes = rag.findEdge(shouldNotEdge)
                assert fRes == -1

            
       
        finally:
            try:
                os.remove(fpath)
                shutil.rmtree(tempFolder)

            except:
                pass
