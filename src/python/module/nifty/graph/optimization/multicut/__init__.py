from __future__ import absolute_import
<<<<<<< HEAD

import sys
=======
from ._multicut import *
from .... import Configuration
from ... import (UndirectedGraph,EdgeContractionGraphUndirectedGraph)
>>>>>>> 935fba7f61ff7020b5515bda2bc1cf2c1c986a36
from functools import partial

from ._multicut import *
from .... import Configuration

__all__ = []
for key in _multicut.__dict__.keys():
    __all__.append(key)



def ilpSettings(relativeGap=0.0, absoluteGap=0.0, memLimit=-1.0):
    s = IlpBackendSettings()
    s.relativeGap = float(relativeGap)
    s.absoluteGap = float(absoluteGap)
    s.memLimit = float(memLimit)

    return s


<<<<<<< HEAD
def __extendMulticutObj(objectiveCls, objectiveName):
=======

def __extendMulticutObj(objectiveCls, objectiveName, graphCls):
>>>>>>> 935fba7f61ff7020b5515bda2bc1cf2c1c986a36


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
    O.verboseVisitor = staticmethod(multicutVerboseVisitor)

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


    def blockMulticut(multicutFactory):
        s,F = getSettingsAndFactoryCls("BlockMulticut")
        s.multicutFactory = multicutFactory
        return F(s)

    O.blockMulticut = staticmethod(blockMulticut)

    def chainedSolversFactory(multicutFactories):
        s,F = getSettingsAndFactoryCls("ChainedSolvers")
        s.multicutFactories = multicutFactories
        return F(s)
    O.chainedSolversFactory = staticmethod(chainedSolversFactory)



    def cgcFactory(doCutPhase=True, doGlueAndCutPhase=True, mincutFactory=None):
        if mincutFactory is None:
            if Configuration.WITH_QPBO:
                mincutFactory = graphCls.MincutObjective.greedyAdditiveFactory(improve=False)
            else:
                raise RuntimeError("default mincutFactory needs to be compiled WITH_QPBO")

        if Configuration.WITH_QPBO:
            s,F = getSettingsAndFactoryCls("Cgc")
            s.doCutPhase = bool(doCutPhase)
            s.doGlueAndCutPhase = bool(doGlueAndCutPhase)
            s.mincutFactory = mincutFactory
            return F(s)
        else:
            raise RuntimeError("cgc need nifty to be compiled WITH_QPBO")
    O.cgcFactory = staticmethod(cgcFactory)



    def defaultMulticutFactory():
        if Configuration.WITH_QPBO:
            return O.cgcFactory()
        else:
            return O.greedyAdditiveFactory()

    O.defaultMulticutFactory = staticmethod(defaultMulticutFactory)


    def multicutAndresGreedyAdditiveFactory():
        s, F = getSettingsAndFactoryCls("MulticutAndresGreedyAdditive")
        return F(s)
    O.multicutAndresGreedyAdditiveFactory = staticmethod(multicutAndresGreedyAdditiveFactory)


    def multicutAndresKernighanLinFactory(
            numberOfInnerIterations = sys.maxsize,
            numberOfOuterIterations = 100,
            epsilon = 1e-6,
            verbose = False,
            greedyWarmstart = True
            ):
        s, F = getSettingsAndFactoryCls("MulticutAndresKernighanLin")
        s.numberOfInnerIterations = numberOfInnerIterations
        s.numberOfOuterIterations = numberOfOuterIterations
        s.epsilon = epsilon
        s.verbose = verbose
        s.greedyWarmstart = greedyWarmstart
        return F(s)
    O.multicutAndresKernighanLinFactory = staticmethod(multicutAndresKernighanLinFactory)


    def multicutDecomposer(submodelFactory=None, fallthroughFactory=None):

        if submodelFactory is None:
           submodelFactory = MulticutObjectiveUndirectedGraph.defaultMulticutFactory()

        if fallthroughFactory is None:
            fallthroughFactory = O.defaultMulticutFactory()


        s,F = getSettingsAndFactoryCls("MulticutDecomposer")
        s.submodelFactory = submodelFactory
        s.fallthroughFactory = fallthroughFactory
        return F(s)

    O.multicutDecomposer = staticmethod(multicutDecomposer)




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


    if Configuration.WITH_LP_MP:
        def multicutMpFactory(
                mcFactory = None,
                greedyWarmstart = False,
                numberOfIterations = 1000,
                verbose = 0,
                primalComputationInterval = 100,
                standardReparametrization = "anisotropic",
                roundingReparametrization = "damped_uniform",
                tightenReparametrization  = "damped_uniform",
                tighten = True,
                tightenInterval = 100,
                tightenIteration = 10,
                tightenSlope = 0.02,
                tightenConstraintsPercentage = 0.1,
                minDualImprovement = 0.,
                minDualImprovementInterval = 0,
                timeout = 0,
                numberOfThreads = 1
                ):

            settings, factoryCls = getSettingsAndFactoryCls("MulticutMp")

            settings.mcFactory = mcFactory
            settings.greedyWarmstart = greedyWarmstart
            settings.numberOfIterations = numberOfIterations
            settings.verbose = verbose
            settings.primalComputationInterval = primalComputationInterval
            settings.standardReparametrization = standardReparametrization
            settings.roundingReparametrization = roundingReparametrization
            settings.tightenReparametrization  = tightenReparametrization
            settings.tighten = tighten
            settings.tightenInterval = tightenInterval
            settings.tightenIteration = tightenIteration
            settings.tightenSlope = tightenSlope
            settings.tightenConstraintsPercentage = tightenConstraintsPercentage
            settings.minDualImprovement = minDualImprovement
            settings.minDualImprovementInterval = minDualImprovementInterval
            settings.timeout = timeout
            settings.numberOfThreads = numberOfThreads

            return factoryCls(settings)

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
    "MulticutObjectiveUndirectedGraph",UndirectedGraph)
__extendMulticutObj(MulticutObjectiveEdgeContractionGraphUndirectedGraph,
    "MulticutObjectiveEdgeContractionGraphUndirectedGraph",EdgeContractionGraphUndirectedGraph)
del __extendMulticutObj
