import nifty
import nifty.graph.opt.ho_multicut as homc
import numpy


import nifty.cgp as ncgp









n_nodes = 5

g = nifty.graph.undirectedGraph(n_nodes)

# add edges
g.insertEdge(0,1)
g.insertEdge(1,2)
g.insertEdge(0,2)
g.insertEdge(2,3)
g.insertEdge(0,4)

w = numpy.random.rand(g.numberOfEdges) - 0.5

# unaries are added direct all at once
obj = homc.hoMulticutObjective(g, w)


potts = numpy.zeros([2,2])
potts[0,1] = 1.0
potts[1,0] = 1.0


obj.addHigherOrderFactor(potts, [0,1])


print("PA")
factory = obj.hoMulticutIlpFactory()
print("PB")
solver = factory.create(obj)
print("PC")
arg = solver.optimize(obj.verboseVisitor())
print("PD")
print(arg)