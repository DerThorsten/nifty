from __future__ import print_function
import nifty
import numpy
import nifty
import nifty.graph
import nifty.graph.rag
import tempfile
import shutil
import os 

nrag = nifty.graph.rag


def ensureDir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)



def genericRagTest(rag, numberOfNodes, shouldEdges, shouldNotEdges):
    assert rag.numberOfNodes == numberOfNodes
    assert rag.numberOfEdges == len(shouldEdges)

    edgeList = []
    for edge in rag.edges():
        edgeList.append(edge)

    assert len(edgeList) == len(shouldEdges)

    for shouldEdge in shouldEdges:

        fRes = rag.findEdge(shouldEdge)
        assert fRes >= 0
        uv = rag.uv(fRes)
        uv = sorted(uv)
        assert uv[0] == shouldEdge[0]
        assert uv[1] == shouldEdge[1]

    for shouldNotEdge in shouldNotEdges:
        fRes = rag.findEdge(shouldNotEdge)
        assert fRes == -1



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

    shouldEdges = [
        (0,1),
        (0,2),
        (0,3),
        (1,2),
        (2,3),
        (2,4),
        (3,4)
    ]

    shouldNotEdges = [
        (0,4),
        (1,3),
        (1,4)
    ]

    genericRagTest(rag=ragA, numberOfNodes=5, 
               shouldEdges=shouldEdges, shouldNotEdges=shouldNotEdges)
    
    genericRagTest(rag=ragB, numberOfNodes=5, 
               shouldEdges=shouldEdges, shouldNotEdges=shouldNotEdges)

    
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

    shouldEdges = [
        (0,1),
        (0,2),
        (0,3),
        (1,2),
        (2,3),
        (2,4),
        (3,4)
    ]

    shouldNotEdges = [
        (0,4),
        (1,3),
        (1,4)
    ]


    genericRagTest(rag=ragA, numberOfNodes=5, 
               shouldEdges=shouldEdges, shouldNotEdges=shouldNotEdges)
    
    genericRagTest(rag=ragB, numberOfNodes=5, 
               shouldEdges=shouldEdges, shouldNotEdges=shouldNotEdges)

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


    shouldEdges = [
        (0,1),
        (0,2),
        (1,2),
        (1,3),
        (2,3)
        
    ]

    shouldNotEdges = [
       (0,3)
    ]

    genericRagTest(rag=ragA, numberOfNodes=4, 
               shouldEdges=shouldEdges, shouldNotEdges=shouldNotEdges)
    
    genericRagTest(rag=ragB, numberOfNodes=4, 
               shouldEdges=shouldEdges, shouldNotEdges=shouldNotEdges)





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


            array[0:shape[0], 0:shape[1]] = labels

            rag = nrag.gridRagHdf5(array, numberOfLabels=labels.max()+1, 
                                   blockShape=blockShape, numberOfThreads=2)

            shouldEdges = [
                (0,1),
                (0,2),
                (1,2),
                (2,3)
            ]

            shouldNotEdges = [
                (0,3),
                (1,3)
            ]

            genericRagTest(rag=rag, numberOfNodes=labels.max()+1, 
                           shouldEdges=shouldEdges, shouldNotEdges=shouldNotEdges)
            
       
        finally:
            try:
                os.remove(fpath)
                shutil.rmtree(tempFolder)

            except:
                pass

    def testHdf5Rag2dLarge():

        tempFolder = tempfile.mkdtemp()
        ensureDir(tempFolder)
        fpath = os.path.join(tempFolder,'_nifty_testHdf5Rag2dLarge_.h5')
        

        try:
            
            shape = [5,6]
            chunkShape = [3,2]
            blockShape =  [2,3]

            hidT = nhdf5.createFile(fpath)
            array = nhdf5.Hdf5ArrayUInt32(hidT, "data", shape, chunkShape)

            assert array.shape[0] == shape[0]
            assert array.shape[1] == shape[1]

            labels = numpy.array( [
                [0, 0, 0, 0, 1, 1],
                [0, 2, 2, 0, 1, 3],
                [0, 3, 3, 3, 3, 3],
                [0, 3, 4, 5, 5, 5],
                [0, 0, 4, 6, 6, 6],
            ],dtype='uint32')

            assert labels.shape[0] == shape[0]
            assert labels.shape[1] == shape[1]


            array[0:shape[0], 0:shape[1]] = labels
            rag = nrag.gridRagHdf5(array, numberOfLabels=labels.max()+1, 
                                   blockShape=blockShape, numberOfThreads=1)


            shouldEdges = [
                (0,1),
                (0,2),
                (0,3),
                (0,4),
                (1,3),
                (2,3),
                (3,4),
                (3,5),
                (4,5),
                (4,6),
                (5,6)
            ]

            shouldNotEdges = [
                (0,6),
                (0,5),
                (1,6),
                (1,5)
            ]

            genericRagTest(rag=rag, numberOfNodes=labels.max()+1, 
                           shouldEdges=shouldEdges, shouldNotEdges=shouldNotEdges)
            
       
        finally:
            try:
                os.remove(fpath)
                shutil.rmtree(tempFolder)

            except:
                pass

    def testHdf5Rag3d():

        tempFolder = tempfile.mkdtemp()
        ensureDir(tempFolder)
        fpath = os.path.join(tempFolder,'_nifty_testHdf5Rag3d_.h5')
        

        try:
            
            shape = [3,2,2]
            chunkShape = [1,2,1]
            blockShape =  [1,2,3]

            hidT = nhdf5.createFile(fpath)
            array = nhdf5.Hdf5ArrayUInt32(hidT, "data", shape, chunkShape)

            assert array.shape[0] == shape[0]
            assert array.shape[1] == shape[1]
            assert array.shape[2] == shape[2]

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
            labels = numpy.array( labels,dtype='uint32')

            assert labels.shape[0] == shape[0]
            assert labels.shape[1] == shape[1]
            assert labels.shape[2] == shape[2]

            array[0:shape[0], 0:shape[1], 0:shape[2]] = labels
            rag = nrag.gridRagHdf5(array, numberOfLabels=labels.max()+1, 
                           blockShape=blockShape, numberOfThreads=-1)


            shouldEdges = [
                (0,1),
                (0,2),
                (1,2),
                (1,3),
                (2,3)
            ]

            shouldNotEdges = [
               (0,3)
            ]

            genericRagTest(rag=rag, numberOfNodes=labels.max()+1, 
                           shouldEdges=shouldEdges, shouldNotEdges=shouldNotEdges)
            
       
        finally:
            try:
                os.remove(fpath)
                shutil.rmtree(tempFolder)

            except:
                pass

    def testGridRag3DStacked2D():

        tempFolder = tempfile.mkdtemp()
        ensureDir(tempFolder)
        fpath = os.path.join(tempFolder,'_nifty_testHdf5Rag3d_.h5')
        

        try:
            
            shape = [3,2,2]
            chunkShape = [1,2,1]
            blockShape =  [1,2,3]

            hidT = nhdf5.createFile(fpath)
            array = nhdf5.Hdf5ArrayUInt32(hidT, "data", shape, chunkShape)

            assert array.shape[0] == shape[0]
            assert array.shape[1] == shape[1]
            assert array.shape[2] == shape[2]

            labels = [
                [
                    [0,1],
                    [0,1]
                ],
                [
                    [2,2],
                    [2,3]
                ],
                [
                    [4,5],
                    [6,6]
                ]
            ]
            labels = numpy.array( labels,dtype='uint32')

            assert labels.shape[0] == shape[0]
            assert labels.shape[1] == shape[1]
            assert labels.shape[2] == shape[2]


            array[0:shape[0], 0:shape[1], 0:shape[2]] = labels
            rag = nrag.gridRagStacked2DHdf5(array, numberOfLabels=labels.max()+1,
                                            numberOfThreads=-1)


            shouldEdges = [
                (0,1),
                (0,2),
                (1,2),
                (1,3),
                (2,3),
                (2,4),
                (2,5),
                (2,6),
                (3,6),
                (4,5),
                (4,6),
                (5,6)
            ]

            shouldNotEdges = [
                (0,3),
                (0,4),
                (0,5),
                (0,6),
                (1,4),
                (1,5),
                (1,6)
            ]


            #("edges in rag",rag.numberOfEdges,len(shouldEdges))

            genericRagTest(rag=rag, numberOfNodes=labels.max()+1, 
                           shouldEdges=shouldEdges, shouldNotEdges=shouldNotEdges)
            
       
        finally:
            try:
                os.remove(fpath)
                shutil.rmtree(tempFolder)

            except:
                pass

    def testGridRag3DStacked2DLarge():

        tempFolder = tempfile.mkdtemp()
        ensureDir(tempFolder)
        fpath = os.path.join(tempFolder,'_nifty_testHdf5Rag3d_.h5')
        

        try:
            
            shape = [3,4,4]
            chunkShape = [1,2,1]


            hidT = nhdf5.createFile(fpath)
            array = nhdf5.Hdf5ArrayUInt32(hidT, "data", shape, chunkShape)

            assert array.shape[0] == shape[0]
            assert array.shape[1] == shape[1]
            assert array.shape[2] == shape[2]

            labels = [
                [
                    [0,0,0,0],
                    [1,1,1,1],
                    [2,2,2,2],
                    [2,2,2,2]
                ],
                [
                    [3,3,3,3],
                    [3,3,3,3],
                    [3,3,3,3],
                    [3,3,3,3]
                ],
                [
                    [4,4,5,5],
                    [4,4,5,5],
                    [4,4,5,5],
                    [4,4,5,5]
                ]
            ]
            labels = numpy.array( labels,dtype='uint32')

            assert labels.shape[0] == shape[0]
            assert labels.shape[1] == shape[1]
            assert labels.shape[2] == shape[2]


            array[0:shape[0], 0:shape[1], 0:shape[2]] = labels
            rag = nrag.gridRagStacked2DHdf5(array, numberOfLabels=labels.max()+1,
                                            numberOfThreads=-1)


            shouldEdges = [
               (0,1),
               (0,3),
               (1,2),
               (1,3),
               (2,3),
               (3,4),
               (3,5),
               (4,5)
            ]

            shouldNotEdges = [
                (0,4),
                (0,5),
                (1,4),
                (1,5),
                (2,4),
                (2,5)
            ]


            genericRagTest(rag=rag, numberOfNodes=labels.max()+1, 
                           shouldEdges=shouldEdges, shouldNotEdges=shouldNotEdges)
            
       
        finally:
            try:
                os.remove(fpath)
                shutil.rmtree(tempFolder)

            except:
                pass

