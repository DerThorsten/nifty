"""
Undirected Graph: Simple Example
=============================================

Very simple example how to use undirected graphs

"""
from __future__ import print_function
import nifty.graph
import numpy
import pylab

##############################################
#  2D undirected grid graph
numberOfNodes = 5
graph = nifty.graph.undirectedGraph(numberOfNodes)
print("#nodes", graph.numberOfNodes)
print("#edges", graph.numberOfEdges)
print(graph)

##############################################
#  insert edges
graph.insertEdge(0,1)
graph.insertEdge(0,2)




##############################################
#  insert multiple edges at once
uvIds = numpy.array([[0,3],[1,2],[1,4],[1,3],[3,4]])
graph.insertEdges(uvIds)
##############################################
# iterate over nodes
# and the adjacency
# of each node
for node in graph.nodes():
    print("u",node)
    for v,e in graph.nodeAdjacency(node):
        print(" v",v,"e",e)


##############################################
# iterate over edges
# and print the endpoints
for edge in graph.edges():
    print("edge ",edge, "uv:", graph.uv(edge))


##############################################
# get the uv-ids /endpoints
# for all edges simultaneous
# as a numpy array
uvIds = graph.uvIds()
print(uvIds)




##############################################
# plot the graph:
# needs networkx
nifty.graph.drawGraph(graph)
pylab.show()