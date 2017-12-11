from __future__ import print_function
import shutil
import os
import unittest
import numpy
import nifty
import nifty.graph.rag as nrag


class TestRagBase(unittest.TestCase):

    def setUp(self):
        self.tmp = './tmp'
        if not os.path.exists(self.tmp):
            os.mkdir(self.tmp)
        self.path = os.path.join(self.tmp, 'rag')
        self.path2 = os.path.join(self.tmp, 'rag2')

    def tearDown(self):
        if os.path.exists(self.tmp):
            shutil.rmtree(self.tmp)

    def generic_rag_test(self, rag, numberOfNodes, shouldEdges, shouldNotEdges, shape=None):
        self.assertEqual(rag.numberOfNodes, numberOfNodes)
        self.assertEqual(rag.numberOfEdges, len(shouldEdges))
        if shape is not None:
            self.assertEqual(rag.shape, shape)

        edgeList = []
        for edge in rag.edges():
            edgeList.append(edge)

        assert len(edgeList) == len(shouldEdges)

        for shouldEdge in shouldEdges:

            fRes = rag.findEdge(shouldEdge)
            self.assertGreaterEqual(fRes, 0)
            uv = rag.uv(fRes)
            uv = sorted(uv)
            self.assertEqual(uv[0], shouldEdge[0])
            self.assertEqual(uv[1], shouldEdge[1])

        for shouldNotEdge in shouldNotEdges:
            fRes = rag.findEdge(shouldNotEdge)
            self.assertEqual(fRes, -1)


# TODO same test skeletons for all rag implementations
class TestRag(TestRagBase):

    # This will fail because the expliecit labels python bindings are broken
    def test_insert(self):
        labels = numpy.zeros(shape=(2, 2), dtype='uint32')

        labels[0, 0] = 0
        labels[1, 0] = 1
        labels[0, 1] = 0
        labels[1, 1] = 2

        g = nrag.gridRag(labels, labels.max() + 1)

        self.assertEqual(g.numberOfNodes, 3)
        self.assertEqual(g.numberOfEdges, 3)

        insertWorked = True
        # TODO we should use a assertRaises here
        try:
            g.insertEdge(0, 1)
        except:
            insertWorked = False
        self.assertFalse(insertWorked)

    # This will fail because the expliecit labels python bindings are broken
    def test_explicit_labels_rag2d(self):
        labels = numpy.array([[0, 1, 2],
                              [0, 0, 2],
                              [3, 3, 2],
                              [4, 4, 4]], dtype='uint32')

        n_labels = labels.max() + 1
        ragA = nifty.graph.rag.gridRag(labels, n_labels, numberOfThreads=1)
        ragB = nifty.graph.rag.gridRag(labels, n_labels)

        self.assertTrue(isinstance(ragA, nifty.graph.rag.ExplicitLabelsGridRag2D))
        self.assertTrue(isinstance(ragB, nifty.graph.rag.ExplicitLabelsGridRag2D))

        shouldEdges = [(0, 1),
                       (0, 2),
                       (0, 3),
                       (1, 2),
                       (2, 3),
                       (2, 4),
                       (3, 4)]

        shouldNotEdges = [(0, 4),
                          (1, 3),
                          (1, 4)]

        self.generic_rag_test(rag=ragA,
                              numberOfNodes=5,
                              shouldEdges=shouldEdges,
                              shouldNotEdges=shouldNotEdges)

        self.generic_rag_test(rag=ragB,
                              numberOfNodes=5,
                              shouldEdges=shouldEdges,
                              shouldNotEdges=shouldNotEdges)

    # This will fail because the expliecit labels python bindings are broken
    def test_explicit_labels_rag3d(self):
        labels = [[[0, 1],
                   [0, 0]],
                  [[1, 1],
                   [2, 2]],
                  [[3, 3],
                   [3, 3]]]
        labels = numpy.array(labels, dtype='uint32')

        n_labels = labels.max() + 1
        ragA = nifty.graph.rag.gridRag(labels, n_labels, numberOfThreads=1)
        ragB = nifty.graph.rag.gridRag(labels, n_labels)

        self.assertTrue(isinstance(ragA, nifty.graph.rag.ExplicitLabelsGridRag3D32))
        self.assertTrue(isinstance(ragB, nifty.graph.rag.ExplicitLabelsGridRag3D32))

        shouldEdges = [(0, 1),
                       (0, 2),
                       (1, 2),
                       (1, 3),
                       (2, 3)]

        shouldNotEdges = [(0, 3)]

        self.generic_rag_test(rag=ragA,
                              numberOfNodes=4,
                              shouldEdges=shouldEdges,
                              shouldNotEdges=shouldNotEdges)

        self.generic_rag_test(rag=ragB,
                              numberOfNodes=4,
                              shouldEdges=shouldEdges,
                              shouldNotEdges=shouldNotEdges)

    @unittest.skipUnless(nifty.Configuration.WITH_HDF5, "skipping hdf5 tests")
    def test_hdf5_rag2d(self):
        import nifty.hdf5 as nhdf5

        shape = [3, 3]
        chunkShape = [1, 1]
        blockShape = [2, 2]

        hidT = nhdf5.createFile(self.path)
        array = nhdf5.Hdf5ArrayUInt32(hidT, "data", shape, chunkShape)

        self.assertEqual(array.shape[0], shape[0])
        self.assertEqual(array.shape[1], shape[1])

        labels = numpy.array([[0, 1, 1],
                              [2, 2, 2],
                              [3, 3, 3]], dtype='uint32')

        self.assertEqual(labels.shape[0], shape[0])
        self.assertEqual(labels.shape[1], shape[1])

        array[0:shape[0], 0:shape[1]] = labels

        rag = nrag.gridRagHdf5(array,
                               numberOfLabels=labels.max() + 1,
                               blockShape=blockShape,
                               numberOfThreads=2)

        shouldEdges = [(0, 1),
                       (0, 2),
                       (1, 2),
                       (2, 3)]

        shouldNotEdges = [(0, 3),
                          (1, 3)]

        self.generic_rag_test(rag=rag,
                              numberOfNodes=labels.max() + 1,
                              shouldEdges=shouldEdges,
                              shouldNotEdges=shouldNotEdges)
        nhdf5.closeFile(hidT)

    @unittest.skipUnless(nifty.Configuration.WITH_HDF5, "skipping hdf5 tests")
    def test_hdf5_rag2d_large(self):
        import nifty.hdf5 as nhdf5

        shape = [5, 6]
        chunkShape = [3, 2]
        blockShape = [2, 3]

        hidT = nhdf5.createFile(self.path)
        array = nhdf5.Hdf5ArrayUInt32(hidT, "data", shape, chunkShape)

        self.assertEqual(array.shape[0], shape[0])
        self.assertEqual(array.shape[1], shape[1])

        labels = numpy.array([[0, 0, 0, 0, 1, 1],
                              [0, 2, 2, 0, 1, 3],
                              [0, 3, 3, 3, 3, 3],
                              [0, 3, 4, 5, 5, 5],
                              [0, 0, 4, 6, 6, 6]],
                             dtype='uint32')

        self.assertEqual(labels.shape[0], shape[0])
        self.assertEqual(labels.shape[1], shape[1])

        array[0:shape[0], 0:shape[1]] = labels
        rag = nrag.gridRagHdf5(array,
                               numberOfLabels=labels.max() + 1,
                               blockShape=blockShape,
                               numberOfThreads=1)

        shouldEdges = [(0, 1),
                       (0, 2),
                       (0, 3),
                       (0, 4),
                       (1, 3),
                       (2, 3),
                       (3, 4),
                       (3, 5),
                       (4, 5),
                       (4, 6),
                       (5, 6)]

        shouldNotEdges = [(0, 6),
                          (0, 5),
                          (1, 6),
                          (1, 5)]

        self.generic_rag_test(rag=rag,
                              numberOfNodes=labels.max() + 1,
                              shouldEdges=shouldEdges,
                              shouldNotEdges=shouldNotEdges)
        nhdf5.closeFile(hidT)

    @unittest.skipUnless(nifty.Configuration.WITH_HDF5, "skipping hdf5 tests")
    def test_hdf5_rag_3d(self):
        import nifty.hdf5 as nhdf5

        shape = [3, 2, 2]
        chunkShape = [1, 2, 1]
        blockShape = [1, 2, 3]

        hidT = nhdf5.createFile(self.path)
        array = nhdf5.Hdf5ArrayUInt32(hidT, "data", shape, chunkShape)

        self.assertEqual(array.shape[0], shape[0])
        self.assertEqual(array.shape[1], shape[1])
        self.assertEqual(array.shape[2], shape[2])

        labels = [[[0, 1],
                   [0, 0]],
                  [[1, 1],
                   [2, 2]],
                  [[3, 3],
                   [3, 3]]]
        labels = numpy.array(labels, dtype='uint32')

        self.assertEqual(labels.shape[0], shape[0])
        self.assertEqual(labels.shape[1], shape[1])
        self.assertEqual(labels.shape[2], shape[2])

        array[0:shape[0], 0:shape[1], 0:shape[2]] = labels
        rag = nrag.gridRagHdf5(array,
                               numberOfLabels=labels.max() + 1,
                               blockShape=blockShape,
                               numberOfThreads=-1)

        shouldEdges = [(0, 1),
                       (0, 2),
                       (1, 2),
                       (1, 3),
                       (2, 3)]

        shouldNotEdges = [(0, 3)]

        self.generic_rag_test(rag=rag,
                              numberOfNodes=labels.max() + 1,
                              shouldEdges=shouldEdges,
                              shouldNotEdges=shouldNotEdges)
        nhdf5.closeFile(hidT)


if __name__ == '__main__':
    unittest.main()
