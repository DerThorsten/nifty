from __future__ import print_function
import nifty
import numpy

import unittest


class TestLiftedMulticutObjective(unittest.TestCase):

    def generateGrid(self, gridSize):
        def nid(x, y):
            return x*gridSize[1] + y
        G = nifty.graph.UndirectedGraph
        g =  G(gridSize[0] * gridSize[1])
        for x in range(gridSize[0]):
            for y in range(gridSize[1]):  

                u = nid(x,y)

                if x + 1 < gridSize[0]:

                    v = nid(x+1, y)
                    g.insertEdge(u, v)

                if y + 1 < gridSize[1]:

                    v = nid(x, y+1)
                    g.insertEdge(u, v)

        return g, nid


    def testInsertLiftedEdgesBfsSmall(self):
        g,nid = self.generateGrid([2,2])


        obj = nifty.graph.lifted_multicut.liftedMulticutObjective(g)
        graph = obj.graph
        liftedGraph = obj.liftedGraph


        self.assertEqual(graph.numberOfEdges,g.numberOfEdges)
        self.assertEqual(liftedGraph.numberOfNodes, graph.numberOfNodes)
        self.assertEqual(liftedGraph.numberOfEdges, graph.numberOfEdges)

        # this should not add any edge at all
        distance = obj.insertLiftedEdgesBfs(1, returnDistance=True)
        postEdges = liftedGraph.numberOfEdges
        self.assertEqual(liftedGraph.numberOfEdges, graph.numberOfEdges)
        self.assertEqual(len(distance),0)

        # this should  add edges
        distance = obj.insertLiftedEdgesBfs(2, returnDistance=True)
        postEdges = liftedGraph.numberOfEdges
        self.assertEqual(liftedGraph.numberOfEdges, 6)
        self.assertEqual(len(distance),2)
        self.assertEqual(distance[0],2)
        self.assertEqual(distance[1],2)

    def testInsertLiftedEdgesBfsBig(self):

        g,nid = self.generateGrid([10, 10])
        obj = nifty.graph.lifted_multicut.liftedMulticutObjective(g)
        graph = obj.graph
        liftedGraph = obj.liftedGraph

        self.assertEqual(graph.numberOfEdges,g.numberOfEdges)
        self.assertEqual(liftedGraph.numberOfNodes, graph.numberOfNodes)
        self.assertEqual(liftedGraph.numberOfEdges, graph.numberOfEdges)

        # this should not add any edge at all
        obj.insertLiftedEdgesBfs(1)
        postEdges = liftedGraph.numberOfEdges
        self.assertEqual(liftedGraph.numberOfEdges, graph.numberOfEdges)

        # this should  add edges
        obj.insertLiftedEdgesBfs(2)
        postEdges = liftedGraph.numberOfEdges

        # check a few lifted edges
        node = nid(0,0)
        self.assertNotEqual( liftedGraph.findEdge(node, nid(0,1)) , -1 )
        self.assertNotEqual( liftedGraph.findEdge(node, nid(0,2)) , -1 )
        self.assertEqual( liftedGraph.findEdge(node, nid(0,3))    , -1 )







class TestLiftedMulticutGridGraphObjective(unittest.TestCase):

    def generateGrid(self, gridSize):
        def nid(x, y):
            return x*gridSize[1] + y
        g = nifty.graph.undirectedGridGraph(gridSize)
    
        return g, nid


    def testInsertLiftedEdgesBfsSmall(self):
        g,nid = self.generateGrid([2,2])


        obj = nifty.graph.lifted_multicut.liftedMulticutObjective(g)
        graph = obj.graph
        liftedGraph = obj.liftedGraph


        self.assertEqual(graph.numberOfEdges,g.numberOfEdges)
        self.assertEqual(liftedGraph.numberOfNodes, graph.numberOfNodes)
        self.assertEqual(liftedGraph.numberOfEdges, graph.numberOfEdges)

        # this should not add any edge at all
        distance = obj.insertLiftedEdgesBfs(1, returnDistance=True)
        postEdges = liftedGraph.numberOfEdges
        self.assertEqual(liftedGraph.numberOfEdges, graph.numberOfEdges)
        self.assertEqual(len(distance),0)

        # this should  add edges
        distance = obj.insertLiftedEdgesBfs(2, returnDistance=True)
        postEdges = liftedGraph.numberOfEdges
        self.assertEqual(liftedGraph.numberOfEdges, 6)
        self.assertEqual(len(distance),2)
        self.assertEqual(distance[0],2)
        self.assertEqual(distance[1],2)

    def testInsertLiftedEdgesBfsBig(self):

        g,nid = self.generateGrid([10, 10])
        obj = nifty.graph.lifted_multicut.liftedMulticutObjective(g)
        graph = obj.graph
        liftedGraph = obj.liftedGraph

        self.assertEqual(graph.numberOfEdges,g.numberOfEdges)
        self.assertEqual(liftedGraph.numberOfNodes, graph.numberOfNodes)
        self.assertEqual(liftedGraph.numberOfEdges, graph.numberOfEdges)

        # this should not add any edge at all
        obj.insertLiftedEdgesBfs(1)
        postEdges = liftedGraph.numberOfEdges
        self.assertEqual(liftedGraph.numberOfEdges, graph.numberOfEdges)

        # this should  add edges
        obj.insertLiftedEdgesBfs(2)
        postEdges = liftedGraph.numberOfEdges

        # check a few lifted edges
        node = nid(0,0)
        self.assertNotEqual( liftedGraph.findEdge(node, nid(0,1)) , -1 )
        self.assertNotEqual( liftedGraph.findEdge(node, nid(0,2)) , -1 )
        self.assertEqual( liftedGraph.findEdge(node, nid(0,3))    , -1 )