"""
Undirected Grid Graph 
=============================================

2D and 3D undirected with simple neighborhood
(4-neighborhood in 2D, 6-neighborhood in 3D)

"""
import nifty.graph


#####################################
#  2D undirected grid graph
shape = [3, 4]
graph = nifty.graph.undirectedGridGraph(shape)
print("#nodes", graph.nodes)
print("#edges", graph.numberOfEdges)
print(graph)

#####################################
#  iterate over nodes / edges

# iterate over nodes
# and the adjacency
# of each node
for node in graph.nodes():
    print("u",node)
    for v,e in graph.nodeAdjacency(node):
        print(" v",v,"e",e)

# iterate over edges
# and print the endpoints
for edge in graph.edges():
    print("edge ",edge, "uv:", graph.uv(edge))


#####################################
#  convenience functions

# get the uv-ids /endpoints
# for all edges simultaneous
# as a numpy array
uvIds = graph.uvIds()
print(uvIds)