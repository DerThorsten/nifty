from __future__ import print_function
from _nifty import *
import types
from functools import partial
import numpy
import time



class Timer:    
    def __init__(self, name=None, verbose=True):
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.dt = self.end - self.start
        if self.verbose:
            if self.name is not None:
                print(self.name,"took",self.dt,"sec") 
            else:
                print("took",self.dt,"sec")




def ilpSettings(relativeGap=0.0, absoluteGap=0.0, memLimit=-1.0):
    s = graph.multicut.IlpBackendSettings()
    s.relativeGap = float(relativeGap)
    s.absoluteGap = float(absoluteGap)
    s.memLimit = float(memLimit)

    return s
    



def __extendMulticutObj(objectiveCls, objectiveName):


    mcMod = graph.multicut

    def getCls(module, prefix, postfix):
        return module.__dict__[prefix+postfix]

    def getSettingsCls(baseName):
        S =  getCls(mcMod, baseName + "Settings" ,objectiveName)
        return S
    def getMcCls(baseName):
        S =  getCls(mcMod, baseName,objectiveName)
        return S
    def getSettings(baseName):
        S =  getSettingsCls(baseName)
        return S()
    def getSettingsAndFactoryCls(baseName):
        s =  getSettings(baseName)
        F =  getCls(mcMod, baseName + "Factory" ,objectiveName)
        return s,F


    O = objectiveCls

    def multicutVerboseVisitor(visitNth=1):
        V = getMcCls("MulticutVerboseVisitor")
        return V(visitNth)
    O.multicutVerboseVisitor = staticmethod(multicutVerboseVisitor)

    def greedyAdditiveProposals(sigma=1.0, weightStopCond=0.0, nodeNumStopCond=-1.0):
        s = getSettings('FusionMoveBasedGreedyAdditiveProposalGen')
        s.sigma = float(sigma)
        s.weightStopCond = float(weightStopCond)
        s.nodeNumStopCond = float(nodeNumStopCond)
        return s
    O.greedyAdditiveProposals = staticmethod(greedyAdditiveProposals)

    def watershedProposals(sigma=1.0, seedFraction=0.0):
        s = getSettings('FusionMoveBasedWatershedProposalGen')
        s.sigma = float(sigma)
        s.seedFraction = float(seedFraction)
        return s
    O.watershedProposals = staticmethod(watershedProposals)

    def greedyAdditiveFactory( weightStopCond=0.0, nodeNumStopCond=-1.0):
        s,F = getSettingsAndFactoryCls("MulticutGreedyAdditive")
        s.weightStopCond = float(weightStopCond)
        s.nodeNumStopCond = float(nodeNumStopCond)
        return F(s)
    O.greedyAdditiveFactory = staticmethod(greedyAdditiveFactory)




    def multicutIlpFactory(verbose=0, addThreeCyclesConstraints=True,
                                addOnlyViolatedThreeCyclesConstraints=True,
                                relativeGap=0.0, absoluteGap=0.0, memLimit=-1.0,
                                ilpSolver = 'cplex'):

        if ilpSolver == 'cplex':
            s,F = getSettingsAndFactoryCls("MulticutIlpCplex")
        elif ilpSolver == 'gurobi':
            s,F = getSettingsAndFactoryCls("MulticutIlpGurobi")
        elif ilpSolver == 'glpk':
            s,F = getSettingsAndFactoryCls("MulticutIlpGlpk")
        else:
            raise RuntimeError("%s is an unknown ilp solver"%str(ilpSolver))
        s.verbose = int(verbose)
        s.addThreeCyclesConstraints = bool(addThreeCyclesConstraints)
        s.addOnlyViolatedThreeCyclesConstraints = bool(addOnlyViolatedThreeCyclesConstraints)
        s.ilpSettings = ilpSettings(relativeGap=relativeGap, absoluteGap=absoluteGap, memLimit=memLimit)
        return F(s)

    O.multicutIlpFactory = staticmethod(multicutIlpFactory)
    O.multicutIlpCplexFactory = staticmethod(partial(multicutIlpFactory,ilpSolver='cplex'))
    O.multicutIlpGurobiFactory = staticmethod(partial(multicutIlpFactory,ilpSolver='gurobi'))
    O.multicutIlpGlpkFactory = staticmethod(partial(multicutIlpFactory,ilpSolver='glpk'))




    def fusionMoveSettings(mcFactory=None):
        if mcFactory is None:
            mcFactory = graph.multicut.MulticutObjectiveUndirectedGraph.greedyAdditiveFactory()
        s = getSettings('FusionMove')
        s.mcFactory = mcFactory
        return s
    O.fusionMoveSettings = staticmethod(fusionMoveSettings)

    def fusionMoveBasedFactory(numberOfIterations=10,verbose=0,
                               numberOfParallelProposals=10, fuseN=2,
                               stopIfNoImprovement=4,
                               numberOfThreads=-1,
                               proposalGen=None,
                               fusionMove=None):
        if proposalGen is None:
            proposalGen = greedyAdditiveProposals()
        if fusionMove is None:
            fusionMove = fusionMoveSettings()
        solverSettings = None



        if isinstance(proposalGen, getSettingsCls("FusionMoveBasedGreedyAdditiveProposalGen") ):
            solverSettings, factoryCls = getSettingsAndFactoryCls("FusionMoveBasedGreedyAdditive")
        elif isinstance(proposalGen, getSettingsCls("FusionMoveBasedWatershedProposalGen") ):
            solverSettings, factoryCls = getSettingsAndFactoryCls("FusionMoveBasedWatershed")
        else:
            raise TypeError(str(proposalGen)+" is of unknown type")

        solverSettings.fusionMoveSettings = fusionMove
        solverSettings.proposalGenSettings = proposalGen
        solverSettings.numberOfIterations = int(numberOfIterations)
        solverSettings.verbose = int(verbose)
        solverSettings.numberOfParallelProposals = int(numberOfParallelProposals)
        solverSettings.fuseN = int(fuseN)
        solverSettings.stopIfNoImprovement = int(stopIfNoImprovement)
        solverSettings.numberOfThreads = int(numberOfThreads)
        factory = factoryCls(solverSettings)
        return factory
    O.fusionMoveBasedFactory = staticmethod(fusionMoveBasedFactory)



    def perturbAndMapSettings(  numberOfIterations=1000,
                                numberOfThreads=-1,
                                verbose=1,
                                noiseType='normal',
                                noiseMagnitude=1.0,
                                mcFactory=None):
        if mcFactory is None:
            mcFactory = fusionMoveBasedFactory(numberOfThreads=0)
        s = getSettings('PerturbAndMap')
        s.numberOfIterations = int(numberOfIterations)
        s.numberOfThreads = int(numberOfThreads)
        s.verbose = int(verbose)
        s.noiseMagnitude = float(noiseMagnitude)

        if(noiseType == 'normal'):
            s.noiseType = getMcCls('PerturbAndMap').NORMAL_NOISE
        elif(noiseType == 'uniform'):
            s.noiseType = getMcCls('PerturbAndMap').UNIFORM_NOISE
        elif(noiseType == 'makeLessCertain'):
            s.noiseType = graph.multicut.PerturbAndMapUndirectedGraph.MAKE_LESS_CERTAIN
        else:
            raise RuntimeError("'%s' is an unknown noise type. Must be 'normal' or 'uniform' or 'makeLessCertain' "%str(noiseType))

        s.mcFactory = mcFactory
        return s
    O.perturbAndMapSettings = staticmethod(perturbAndMapSettings)

    def perturbAndMap(objective, settings):
        pAndM = graph.multicut.perturbAndMap(objective, settings)
        return pAndM
    O.perturbAndMap = staticmethod(perturbAndMap)


__extendMulticutObj(graph.multicut.MulticutObjectiveUndirectedGraph, 
    "MulticutObjectiveUndirectedGraph")
__extendMulticutObj(graph.multicut.MulticutObjectiveEdgeContractionGraphUndirectedGraph, 
    "MulticutObjectiveEdgeContractionGraphUndirectedGraph")
del __extendMulticutObj



# multicut objective
graph.UndirectedGraph.MulticutObjective = graph.multicut.MulticutObjectiveUndirectedGraph
graph.UndirectedGraph.EdgeContractionGraph = graph.EdgeContractionGraphUndirectedGraph
graph.EdgeContractionGraphUndirectedGraph.MulticutObjective = graph.multicut.MulticutObjectiveEdgeContractionGraphUndirectedGraph





# lifted multicut objective
graph.UndirectedGraph.LiftedMulticutObjective = graph.lifted_multicut.LiftedMulticutObjectiveUndirectedGraph

def __extendLiftedMulticutObj(objectiveCls, objectiveName):
    
    def insertLiftedEdgesBfs(self, maxDistance, returnDistance = False):
        if returnDistance :
            return self._insertLiftedEdgesBfsReturnDist(maxDistance)
        else:
            self._insertLiftedEdgesBfs(maxDistance)

    objectiveCls.insertLiftedEdgesBfs = insertLiftedEdgesBfs





    lmcMod = graph.lifted_multicut

    def getCls(module, prefix, postfix):
        return module.__dict__[prefix+postfix]

    def getSettingsCls(baseName):
        S =  getCls(lmcMod, baseName + "Settings" ,objectiveName)
        return S
    def getLmcCls(baseName):
        S =  getCls(lmcMod, baseName,objectiveName)
        return S
    def getSettings(baseName):
        S =  getSettingsCls(baseName)
        return S()
    def getSettingsAndFactoryCls(baseName):
        s =  getSettings(baseName)
        F =  getCls(lmcMod, baseName + "Factory" ,objectiveName)
        return s,F


    O = objectiveCls

    def verboseVisitor(visitNth=1):
        V = getLmcCls("LiftedMulticutVerboseVisitor")
        return V(visitNth)
    O.verboseVisitor = staticmethod(verboseVisitor)




    def liftedMulticutGreedyAdditiveFactory( weightStopCond=0.0, nodeNumStopCond=-1.0):
        s,F = getSettingsAndFactoryCls("LiftedMulticutGreedyAdditive")
        s.weightStopCond = float(weightStopCond)
        s.nodeNumStopCond = float(nodeNumStopCond)
        return F(s)
    O.liftedMulticutGreedyAdditiveFactory = staticmethod(liftedMulticutGreedyAdditiveFactory)


    def liftedMulticutKernighanLinFactory( numberOfOuterIterations=1000000,
                                            numberOfInnerIterations=100,
                                            epsilon=1e-7):
        s,F = getSettingsAndFactoryCls("LiftedMulticutKernighanLin")
        s.numberOfOuterIterations = int(numberOfOuterIterations)
        s.numberOfInnerIterations = int(numberOfInnerIterations)
        s.epsilon = float(epsilon)
        return F(s)
    O.liftedMulticutKernighanLinFactory = staticmethod(liftedMulticutKernighanLinFactory)


    def liftedMulticutAndresKernighanLinFactory( numberOfOuterIterations=1000000,
                                            numberOfInnerIterations=100,
                                            epsilon=1e-7):
        s,F = getSettingsAndFactoryCls("LiftedMulticutAndresKernighanLin")
        s.numberOfOuterIterations = int(numberOfOuterIterations)
        s.numberOfInnerIterations = int(numberOfInnerIterations)
        s.epsilon = float(epsilon)
        return F(s)
    O.liftedMulticutAndresKernighanLinFactory = staticmethod(liftedMulticutAndresKernighanLinFactory)


    def liftedMulticutAndresGreedyAdditiveFactory():
        s,F = getSettingsAndFactoryCls("LiftedMulticutAndresGreedyAdditive")
        return F(s)
    O.liftedMulticutAndresGreedyAdditiveFactory = staticmethod(liftedMulticutAndresGreedyAdditiveFactory)





    def liftedMulticutIlpFactory(verbose=0, addThreeCyclesConstraints=True,
                                addOnlyViolatedThreeCyclesConstraints=True,
                                relativeGap=0.0, absoluteGap=0.0, memLimit=-1.0,
                                ilpSolver = 'cplex'):

        if ilpSolver == 'cplex':
            s,F = getSettingsAndFactoryCls("LiftedMulticutIlpCplex")
        elif ilpSolver == 'gurobi':
            s,F = getSettingsAndFactoryCls("LiftedMulticutIlpGurobi")
        elif ilpSolver == 'glpk':
            s,F = getSettingsAndFactoryCls("LiftedMulticutIlpGlpk")
        else:
            raise RuntimeError("%s is an unknown ilp solver"%str(ilpSolver))
        s.verbose = int(verbose)
        s.addThreeCyclesConstraints = bool(addThreeCyclesConstraints)
        s.addOnlyViolatedThreeCyclesConstraints = bool(addOnlyViolatedThreeCyclesConstraints)
        s.ilpSettings = ilpSettings(relativeGap=relativeGap, absoluteGap=absoluteGap, memLimit=memLimit)
        return F(s)

    O.liftedMulticutIlpFactory = staticmethod(liftedMulticutIlpFactory)
    O.liftedMulticutIlpCplexFactory = staticmethod(partial(liftedMulticutIlpFactory,ilpSolver='cplex'))
    O.liftedMulticutIlpGurobiFactory = staticmethod(partial(liftedMulticutIlpFactory,ilpSolver='gurobi'))
    O.liftedMulticutIlpGlpkFactory = staticmethod(partial(liftedMulticutIlpFactory,ilpSolver='glpk'))






__extendLiftedMulticutObj(graph.UndirectedGraph.LiftedMulticutObjective, 
    "LiftedMulticutObjectiveUndirectedGraph")
del __extendLiftedMulticutObj







def __extendEdgeContractionGraph():


    class EdgeContractionGraphCallback(graph.EdgeContractionGraphCallbackImpl):
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

    EdgeContractionGraphCallback.__module__ = "nifty.graph"
    graph.EdgeContractionGraphCallback = EdgeContractionGraphCallback
    
    def edgeContractionGraph(g, callback):
        Ecg = g.__class__.EdgeContractionGraph
        ecg = Ecg(g, callback)
        return ecg 

    edgeContractionGraph.__module__ = "nifty.graph"
    graph.edgeContractionGraph = edgeContractionGraph

__extendEdgeContractionGraph()
del __extendEdgeContractionGraph






def __addStaticMethodsToUndirectedGraph():




    G = graph.UndirectedGraph
    CG = G.EdgeContractionGraph

    def _getGalaContractionOrderSettings(
        mcMapFactory=CG.MulticutObjective.fusionMoveBasedFactory(),
        runMcMapEachNthTime=1
    ):
        s =  graph.gala.GalaContractionOrderSettingsUndirectedGraph()
        s.mcMapFactory = mcMapFactory
        s.runMcMapEachNthTime = int(runMcMapEachNthTime)
        return s

    G.galaContractionOrderSettings = staticmethod(_getGalaContractionOrderSettings)


    def _getGalaSettings(threshold0=0.1, threshold1=0.9, thresholdU=0.1, numberOfEpochs=3, numberOfTrees=100,
                         contractionOrderSettings = G.galaContractionOrderSettings(),
                         mapFactory=G.MulticutObjective.fusionMoveBasedFactory(), 
                         perturbAndMapFactory=G.MulticutObjective.fusionMoveBasedFactory()):
        s =  graph.gala.GalaSettingsUndirectedGraph()
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
        return graph.gala.GalaUndirectedGraph(settings)
    G.gala = staticmethod(_getGala)




__addStaticMethodsToUndirectedGraph()
del __addStaticMethodsToUndirectedGraph



def __extendRag():

    def gridRag(labels, numberOfThreads=-1):
        labels = numpy.require(labels)
        if numpy.squeeze(labels).ndim == 2:
            return graph.rag.explicitLabelsGridRag2D(labels, numberOfThreads=int(numberOfThreads))
        elif numpy.squeeze(labels).ndim == 3:
            return graph.rag.explicitLabelsGridRag3D(labels, numberOfThreads=int(numberOfThreads))
        else:
            raise RuntimeError("wrong dimension, currently only 2D and 3D is implemented")

    gridRag.__module__ = "nifty.graph.rag"
    graph.rag.gridRag = gridRag


    if Configuration.WITH_HDF5:

        def gridRagHdf5(labels, numberOfLabels, blockShape = None, numberOfThreads=-1):

            dim = labels.ndim
            if blockShape is None:
                bs = [100]*dim
            else:
                bs = blockShape

            if dim == 2:
                labelsProxy = graph.rag.gridRag2DHdf5LabelsProxy(labels, int(numberOfLabels))
                rag = graph.rag.gridRag2DHdf5(labelsProxy,bs,int(numberOfThreads))
            elif dim == 3:
                labelsProxy = graph.rag.gridRag3DHdf5LabelsProxy(labels, int(numberOfLabels))
                rag = graph.rag.gridRag3DHdf5(labelsProxy,bs,int(numberOfThreads))
            else:
                raise RuntimeError("gridRagHdf5 is only implemented for 2D and 3D not for %dD"%dim)

            return rag

        gridRagHdf5.__module__ = "nifty.graph.rag"
        graph.rag.gridRagHdf5 = gridRagHdf5

        def gridRagStacked2DHdf5(labels, numberOfLabels, numberOfThreads=-1):
            dim = labels.ndim
            if dim == 3:
                labelsProxy = graph.rag.gridRag3DHdf5LabelsProxy(labels, int(numberOfLabels))
                rag = graph.rag.gridRagStacked2DHdf5Impl(labelsProxy,int(numberOfThreads))
            else:
                raise RuntimeError("gridRagStacked2DHdf5 is only implemented for 3D not for %dD"%dim)

            return rag

        gridRagStacked2DHdf5.__module__ = "nifty.graph.rag"
        graph.rag.gridRagStacked2DHdf5 = gridRagStacked2DHdf5


__extendRag()
del __extendRag


if Configuration.WITH_HDF5:

    def __extendHdf5():
        hdf5Arrays = [
            hdf5.Hdf5ArrayUInt8,
            hdf5.Hdf5ArrayUInt16,
            hdf5.Hdf5ArrayUInt32,
            hdf5.Hdf5ArrayUInt64,
            hdf5.Hdf5ArrayInt8,
            hdf5.Hdf5ArrayInt16,
            hdf5.Hdf5ArrayInt32,
            hdf5.Hdf5ArrayInt64,
            hdf5.Hdf5ArrayFloat32,
            hdf5.Hdf5ArrayFloat64
        ]

        def getItem(self, slicing):
            dim = self.ndim
            roiBegin = [None]*dim
            roiEnd = [None]*dim
            for d in range(dim):
                sliceObj = slicing[d]
                roiBegin[d] = int(sliceObj.start)
                roiEnd[d] = int(sliceObj.stop)
                step = sliceObj.step
                if step is not None and  step != 1:
                    raise RuntimeError("currently step must be 1 in slicing but step is %d"%sliceObj.step)

            return self.readSubarray(roiBegin, roiEnd)

        def setItem(self, slicing, value):
            asArray = numpy.require(value)

            dim = self.ndim
            roiBegin = [None]*dim
            roiEnd = [None]*dim
            shape = [None]*dim
            for d in range(dim):
                sliceObj = slicing[d]
                roiBegin[d] = int(sliceObj.start)
                roiEnd[d] = int(sliceObj.stop)
                if roiEnd[d] - roiBegin[d] != asArray.shape[d]:
                    raise RuntimeError("array to write does not match slicing shape")
                step = sliceObj.step
                if step is not None and  step != 1:
                    raise RuntimeError("currently step must be 1 in slicing but step is %d"%sliceObj.step)

            return self.writeSubarray(roiBegin, asArray)


        for array in hdf5Arrays:
            array.__getitem__ = getItem
            array.__setitem__ = setItem






    __extendHdf5()
    del __extendHdf5
