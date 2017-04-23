from __future__ import print_function
import nifty

import nifty.graph
import numpy

def graphAndWeights():
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


def testShortestPathDijkstraSingleTarget():
    g, weights = graphAndWeights()
    sp = nifty.graph.ShortestPathDijkstra(g)
    path = sp.runSingleSourceSingleTarget(weights, 0, 4)

    # shortest path 0 -> 4:
    # 0 - 1, 1 - 2, 3 - 4
    assert len(path) == 5, str(len(path))
    path.reverse()
    for ii in range(5):
        assert path[ii] == ii, "%i, %i" % (path[ii], ii)
    print("Test Single Target successfull")


def testShortestPathDijkstraMultiTarget():
    g, weights = graphAndWeights()
    sp = nifty.graph.ShortestPathDijkstra(g)
    paths = sp.runSingleSourceMultiTarget(weights, 0, [4,5])

    assert len(paths) == 2

    # shortest path 0 -> 4:
    # 0 - 1, 1 - 2, 3 - 4
    path = paths[0]
    path.reverse()
    assert len(path) == 5, str(len(path))
    for ii in range(5):
        assert path[ii] == ii, "%i, %i" % (path[ii], ii)

    # shortest path 0 -> 5:
    # 0 - 5
    path = paths[1]
    path.reverse()
    assert len(path) == 2, str(len(path))
    assert path[0] == 0, str(path[0])
    assert path[1] == 5, str(path[1])
    print("Test Multi Target successfull")


# TODO check that invalid paths are handled correctly
if __name__ == '__main__':
    testShortestPathDijkstraSingleTarget()
    testShortestPathDijkstraMultiTarget()
