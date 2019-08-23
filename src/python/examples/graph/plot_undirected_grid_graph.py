"""
Undirected Grid Graph 
=============================================

2D and 3D undirected with simple neighborhood
(4-neighborhood in 2D, 6-neighborhood in 3D)

"""
from __future__ import print_function
import nifty.graph
import pylab

##############################################
#  2D undirected grid graph
shape = [3, 3]
graph = nifty.graph.undirectedGridGraph(shape)
print("#nodes", graph.numberOfNodes)
print("#edges", graph.numberOfEdges)
print(graph)

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
# get the coordinates of a node
for node in graph.nodes():
    print("node",node,"coordiante",graph.nodeToCoordinate(node))


##############################################
# get the node of a coordinate
for x0 in range(shape[0]):
    for x1 in range(shape[1]):
        print("coordiante",[x0,x1],"node",graph.coordinateToNode([x0,x1]))


##############################################
# plot the graph:
# needs networkx
nifty.graph.drawGraph(graph)
pylab.show()