import nifty
import nifty.graph.opt.ho_multicut as homc
import numpy


import nifty.cgp as ncgp








shape = [5,5]
n_nodes = shape[0] * shape[1]

g = nifty.graph.undirectedGraph(n_nodes)

# add edges
def vi(x,y):
    return y + x*shape[1]

for x in range(shape[0]):
    for y in range(shape[1]):
        if x + 1 < shape[0]:
            g.insertEdge(vi(x,y), vi(x+1,y))
        if y + 1 < shape[1]:
            g.insertEdge(vi(x,y), vi(x,y+1))

w = numpy.random.rand(g.numberOfEdges) - 0.5

# unaries are added direct all at once
obj = homc.hoMulticutObjective(g, w)


potts = numpy.zeros([2,2])
potts[0,1] = 1.0
potts[1,0] = 1.0


obj.addHigherOrderFactor(potts, [0,1])



factory = obj.hoMulticutIlpFactory()
solver = factory.create(obj)
arg = solver.optimize(obj.verboseVisitor())
print(arg)