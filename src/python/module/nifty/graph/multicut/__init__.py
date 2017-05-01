from __future__ import absolute_import
from ._multicut import *
from functools import partial

__all__ = []
for key in _multicut.__dict__.keys():
    __all__.append(key)



def ilpSettings(relativeGap=0.0, absoluteGap=0.0, memLimit=-1.0):
    s = IlpBackendSettings()
    s.relativeGap = float(relativeGap)
    s.absoluteGap = float(absoluteGap)
    s.memLimit = float(memLimit)

    return s


def mpSettings(
    primalComputationInterval = 100,
    standardReparametrization = "anisotropic",
    roundingReparametrization = "damped_uniform",
    tightenReparametrization  = "damped_uniform",
    tighten = True,
    tightenInterval = 100,
    tightenIteration = 2,
    tightenSlope = 0.05,
    tightenConstraintsPercentage = 0.1,
    maxIter = 1000,
    minDualImprovement = 0.,
    minDualImprovementInterval = 0,
    timeout = 0
    ):

    s = MpSettings()

    s.primalComputationInterval = primalComputationInterval
    s.standardReparametrization = standardReparametrization
    s.roundingReparametrization = roundingReparametrization
    s.tightenReparametrization  = tightenReparametrization
    s.tighten = tighten
    s.tightenInterval = tightenInterval
    s.tightenIteration = tightenIteration
    s.tightenSlope = tightenSlope
    s.tightenConstraintsPercentage = tightenConstraintsPercentage
    s.maxIter = maxIter
    s.minDualImprovement = minDualImprovement
    s.minDualImprovementInterval = minDualImprovementInterval
    s.timeout = timeout

    return s



def __extendMulticutObj(objectiveCls, objectiveName):



    def getCls(prefix, postfix):
        return _multicut.__dict__[prefix+postfix]

    def getSettingsCls(baseName):
        S =  getCls(baseName + "Settings" ,objectiveName)
        return S
    def getMcCls(baseName):
        S =  getCls(baseName,objectiveName)
        return S
    def getSettings(baseName):
        S =  getSettingsCls(baseName)
        return S()
    def getSettingsAndFactoryCls(baseName):
        s =  getSettings(baseName)
        F =  getCls(baseName + "Factory" ,objectiveName)
        return s,F


    O = objectiveCls

    def multicutVerboseVisitor(visitNth=1,timeLimit=0):
        V = getMcCls("MulticutVerboseVisitor")
        return V(visitNth,timeLimit)
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


    def multicutMpFactory( mpSettings=mpSettings() ):
        solver = MulticutMp()
        solver.mpSettings = mpSettings
        return solver
    O.multicutMpFactory = staticmethod(multicutMpFactory)


    def fusionMoveSettings(mcFactory=None):
        if mcFactory is None:
            mcFactory = MulticutObjectiveUndirectedGraph.greedyAdditiveFactory()
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


__extendMulticutObj(MulticutObjectiveUndirectedGraph,
    "MulticutObjectiveUndirectedGraph")
__extendMulticutObj(MulticutObjectiveEdgeContractionGraphUndirectedGraph,
    "MulticutObjectiveEdgeContractionGraphUndirectedGraph")
del __extendMulticutObj
