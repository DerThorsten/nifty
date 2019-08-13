from __future__ import print_function
import nifty
import nifty
import numpy
import unittest


class TestUndirectedGridGraph(unittest.TestCase):

    def test2DSimpleNh(self):

        shape = [3,4]
        g =  nifty.graph.undirectedGridGraph(shape)

        self.assertEqual(g.numberOfNodes,12)

        def vi(x0, x1):
            return x0*shape[1] + x1

        counter = 0
        eCounter = 0
        for x0 in range(shape[0]):
            for x1 in range(shape[1]):

                u = vi(x0, x1)
                self.assertEqual(u, counter)
                coord = g.nodeToCoordinate(u)

                self.assertEqual(g.coordinateToNode(coord),u)
                self.assertEqual(g.coordinateToNode(coord),counter)
                self.assertEqual(g.coordinateToNode([x0,x1]),counter)


                self.assertEqual(coord[0], x0)
                self.assertEqual(coord[1], x1)

                if x0 + 1 < shape[0]:
                    v = vi(x0 + 1, x1)
                    e = g.findEdge(u,v)
                    uu,vv = g.uv(e)
                    self.assertNotEqual(uu,vv)
                    self.assertIn(uu,[u,v])
                    self.assertIn(vv,[u,v])
                    self.assertGreater(e,-1)
                    eCounter += 1
                if x1 + 1 < shape[1]:
                    v = vi(x0, x1 + 1)
                    e = g.findEdge(u,v)
                    uu,vv = g.uv(e)
                    self.assertNotEqual(uu,vv)
                    self.assertIn(uu,[u,v])
                    self.assertIn(vv,[u,v])
                    self.assertGreater(e,-1)
                    eCounter += 1
                counter += 1

        self.assertEqual(eCounter,g.numberOfEdges)
        for node in g.nodes():
            coord = g.nodeToCoordinate(node)
            self.assertEqual(g.coordinateToNode(coord),node)

        for edge in g.nodes():
            uv = g.uv(edge)


if __name__ == '__main__':
    unittest.main()
