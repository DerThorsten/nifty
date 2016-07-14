from _nifty import *

from functools import partial







def __extendObj(objectiveCls, objectiveName):


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

    def greedyAdditiveFactory(verbose=0):
        s,F = getSettingsAndFactoryCls("MulticutGreedyAdditive")
        s.verbose = int(verbose)
        return F(s)
    O.greedyAdditiveFactory = staticmethod(greedyAdditiveFactory)


    def ilpSettings(relativeGap=0.0, absoluteGap=0.0, memLimit=-1.0):
        s = graph.multicut.IlpBackendSettings()
        s.relativeGap = float(relativeGap)
        s.absoluteGap = float(absoluteGap)
        s.memLimit = float(memLimit)

        return s
    O.ilpSettings = staticmethod(ilpSettings)


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


__extendObj(graph.multicut.MulticutObjectiveUndirectedGraph, 
    "MulticutObjectiveUndirectedGraph")
__extendObj(graph.multicut.MulticutObjectiveEdgeContractionGraphUndirectedGraph, 
    "MulticutObjectiveEdgeContractionGraphUndirectedGraph")
del __extendObj












graph.UndirectedGraph.MulticutObjective = graph.multicut.MulticutObjectiveUndirectedGraph
graph.EdgeContractionGraphUndirectedGraph.MulticutObjective = graph.multicut.MulticutObjectiveEdgeContractionGraphUndirectedGraph









def __addStaticMethodsToUndirectedGraph():




    G = graph.UndirectedGraph
    def _getGalaSettings(threshold0=0.1, threshold1=0.9, thresholdU=0.1, numberOfEpochs=3, numberOfTrees=100,
                         mapFactory=G.MulticutObjective.fusionMoveBasedFactory(), 
                         perturbAndMapFactory=G.MulticutObjective.fusionMoveBasedFactory()):
        s =  graph.gala.GalaSettingsUndirectedGraph()
        s.threshold0 = float(threshold0)
        s.threshold1 = float(threshold1)
        s.thresholdU = float(thresholdU)
        s.numberOfEpochs = int(numberOfEpochs)
        s.numberOfTrees = int(numberOfTrees)
        s.mapFactory = mapFactory
        s.perturbAndMapFactory = perturbAndMapFactory
        return s

    G.galaSettings = staticmethod(_getGalaSettings)


    G = graph.UndirectedGraph
    def _getGala(settings = G.galaSettings()):
        return graph.gala.GalaUndirectedGraph(settings)
    G.gala = staticmethod(_getGala)


__addStaticMethodsToUndirectedGraph()
del __addStaticMethodsToUndirectedGraph



def __extendRag():

    def gridRag(labels, numberOfThreads=-1, lockFreeAlg=False):
        if labels.ndim == 2:
            return graph.rag.explicitLabelsGridRag2D(labels, numberOfThreads=int(numberOfThreads),
                                           lockFreeAlg=bool(lockFreeAlg))
        elif labels.ndim == 3:
            return graph.rag.explicitLabelsGridRag2D(labels, numberOfThreads=int(numberOfThreads),
                                           lockFreeAlg=bool(lockFreeAlg))
        else:
            raise RuntimeError("wrong dimension, currently only 2D and 3D is implemented")

    gridRag.__module__ = "nifty.graphs.rag"
    graph.rag.gridRag = gridRag

__extendRag()
del __extendRag
