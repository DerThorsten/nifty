from __future__ import print_function
import unittest
import numpy
import nifty
import nifty.graph


class TestShortesetPath(unittest.TestCase):
    def graphAndWeights(self):
        edges = numpy.array([
            [0,1],
            [1,2],
            [2,3],
            [3,4],
            [0,5],
            [4,5]
        ], dtype = 'uint64')
        g = nifty.graph.UndirectedGraph(6)
        g.insertEdges(edges)

        weights = numpy.zeros(len(edges), dtype = 'float32')
        weights[0]  = 1.
        weights[1]  = 1.
        weights[-1] = 5.

        return g, weights

    def testShortestPathDijkstraSingleTarget(self):
        g, weights = self.graphAndWeights()
        sp = nifty.graph.ShortestPathDijkstra(g)
        path = sp.runSingleSourceSingleTarget(weights, 0, 4)
        # shortest path 0 -> 4:
        # 0 - 1, 1 - 2, 3 - 4
        self.assertEqual(len(path), 5, str(len(path)))
        path.reverse()
        for ii in range(5):
            self.assertEqual(path[ii], ii, "%i, %i" % (path[ii], ii))

    def testShortestPathDijkstraSingleTargetParallel(self):
        g, weights = self.graphAndWeights()
        N = 50
        sources = N * [0]
        targets = N * [4]
        parallelPaths = nifty.graph.shortestPathSingleTargetParallel(
            g,
            weights.tolist(),
            sources,
            targets,
            returnNodes=True,
            numberOfThreads=4
        )

        for path in parallelPaths:
            # shortest path 0 -> 4:
            # 0 - 1, 1 - 2, 3 - 4
            self.assertEqual(len(path), 5, str(len(path)))
            path.reverse()
            for ii in range(5):
                self.assertEqual(path[ii], ii, "%i, %i" % (path[ii], ii))

    def testShortestPathDijkstraMultiTarget(self):
        g, weights = self.graphAndWeights()
        sp = nifty.graph.ShortestPathDijkstra(g)
        # we need to check 2 times to make sure that more than 1 runs work
        for _ in range(2):
            paths = sp.runSingleSourceMultiTarget(weights, 0, [4,5])
            self.assertEqual(len(paths), 2)
            # shortest path 0 -> 4:
            # 0 - 1, 1 - 2, 3 - 4
            path = paths[0]
            path.reverse()
            self.assertEqual(len(path), 5, str(len(path)))
            for ii in range(5):
                self.assertEqual(path[ii], ii, "%i, %i" % (path[ii], ii))

            # shortest path 0 -> 5:
            # 0 - 5
            path = paths[1]
            path.reverse()
            self.assertEqual(len(path) , 2, str(len(path)))
            self.assertEqual(path[0] , 0, str(path[0]))
            self.assertEqual(path[1] , 5, str(path[1]))

    def testShortestPathDijkstraMultiTargetParallel(self):
        g, weights = self.graphAndWeights()
        N = 50
        sources = N*[0]
        targets = [[4,5] for _ in range(N)]

        for _ in range(2):
            parallelPaths = nifty.graph.shortestPathMultiTargetParallel(
                g,
                weights,
                sources,
                targets,
                returnNodes=True,
                numberOfThreads=5
            )
            for paths in parallelPaths:
                self.assertEqual(len(paths), 2)
                # shortest path 0 -> 4:
                # 0 - 1, 1 - 2, 3 - 4
                path = paths[0]
                path.reverse()
                self.assertEqual(len(path), 5, str(len(path)))
                for ii in range(5):
                    self.assertEqual(path[ii] , ii, "%i, %i" % (path[ii], ii))

                # shortest path 0 -> 5:
                # 0 - 5
                path = paths[1]
                path.reverse()
                self.assertEqual(len(path) , 2, str(len(path)))
                self.assertEqual(path[0] , 0, str(path[0]))
                self.assertEqual(path[1] , 5, str(path[1]))

    def testShortestPathInvalid(self):
        edges = numpy.array([
            [0,1],
            [2,3]
        ], dtype = 'uint64')
        g = nifty.graph.UndirectedGraph(4)
        g.insertEdges(edges)
        sp = nifty.graph.ShortestPathDijkstra(g)
        weights = [1.,1.]
        path = sp.runSingleSourceSingleTarget(weights, 0, 3)
        self.assertTrue(not path) # make sure that the path is invalid
