from _graph import *
from .. import Configuration

import multicut
import lifted_multicut
import gala

import numpy
from functools import partial
import types

__all__ = []

for key in _graph.__dict__.keys():
    __all__.append(key)


ilpSettings = multicut.ilpSettings





# multicut objective
UndirectedGraph.MulticutObjective = multicut.MulticutObjectiveUndirectedGraph
UndirectedGraph.EdgeContractionGraph = EdgeContractionGraphUndirectedGraph
EdgeContractionGraphUndirectedGraph.MulticutObjective = multicut.MulticutObjectiveEdgeContractionGraphUndirectedGraph





# lifted multicut objective
UndirectedGraph.LiftedMulticutObjective = lifted_multicut.LiftedMulticutObjectiveUndirectedGraph


UndirectedGraph.WeightedLiftedMulticutObjective = lifted_multicut.WeightedLiftedMulticutObjectiveUndirectedGraph





class EdgeContractionGraphCallback(EdgeContractionGraphCallbackImpl):
    def __init__(self):
        super(EdgeContractionGraphCallback, self).__init__()

        
        try:
            self.contractEdgeCallback = types.MethodType(self.contractEdge, self, 
                                            EdgeContractionGraphCallback)
        except AttributeError:
            pass

        try:    
            self.mergeEdgesCallback = types.MethodType(self.mergeEdges, self, 
                                            EdgeContractionGraphCallback)
        except AttributeError:
            pass

        try:
            self.mergeNodesCallback = types.MethodType(self.mergeNodes, self, 
                                        EdgeContractionGraphCallback)
        except AttributeError:
            pass

        try:
            self.contractEdgeDoneCallback = types.MethodType(self.contractEdgeDone, self, 
                                        EdgeContractionGraphCallback)
        except AttributeError:
            pass

    #def contractEdgeCallback(self, edge):
    #    pass
    #def contractEdgeDoneCallback(self, edge):
    #    pass

#EdgeContractionGraphCallback.__module__ = "graph"
EdgeContractionGraphCallback = EdgeContractionGraphCallback

def edgeContractionGraph(g, callback):
    Ecg = g.__class__.EdgeContractionGraph
    ecg = Ecg(g, callback)
    return ecg 







def __addStaticMethodsToUndirectedGraph():




    G = UndirectedGraph
    CG = G.EdgeContractionGraph

    def _getGalaContractionOrderSettings(
        mcMapFactory=CG.MulticutObjective.fusionMoveBasedFactory(),
        runMcMapEachNthTime=1
    ):
        s =  gala.GalaContractionOrderSettingsUndirectedGraph()
        s.mcMapFactory = mcMapFactory
        s.runMcMapEachNthTime = int(runMcMapEachNthTime)
        return s

    G.galaContractionOrderSettings = staticmethod(_getGalaContractionOrderSettings)


    def _getGalaSettings(threshold0=0.1, threshold1=0.9, thresholdU=0.1, numberOfEpochs=3, numberOfTrees=100,
                         contractionOrderSettings = G.galaContractionOrderSettings(),
                         mapFactory=G.MulticutObjective.fusionMoveBasedFactory(), 
                         perturbAndMapFactory=G.MulticutObjective.fusionMoveBasedFactory()):
        s =  gala.GalaSettingsUndirectedGraph()
        s.threshold0 = float(threshold0)
        s.threshold1 = float(threshold1)
        s.thresholdU = float(thresholdU)
        s.numberOfEpochs = int(numberOfEpochs)
        s.numberOfTrees = int(numberOfTrees)
        s.contractionOrderSettings = contractionOrderSettings
        s.mapFactory = mapFactory
        s.perturbAndMapFactory = perturbAndMapFactory
        return s

    G.galaSettings = staticmethod(_getGalaSettings)



    def _getGala(settings = G.galaSettings()):
        return gala.GalaUndirectedGraph(settings)
    G.gala = staticmethod(_getGala)




__addStaticMethodsToUndirectedGraph()
del __addStaticMethodsToUndirectedGraph

