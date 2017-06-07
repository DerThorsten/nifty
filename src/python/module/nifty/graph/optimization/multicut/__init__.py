""" Multicut module of nifty    

This module implements multicut
related functionality.

For more details, see seection :ref:`theory_multicut`.

"""



from __future__ import absolute_import
import sys
from functools import partial
from . import _multicut as __multicut
from ._multicut import *
from .... import Configuration
from ... import (UndirectedGraph,EdgeContractionGraphUndirectedGraph)

__all__ = [
    "ilpSettings"
]
for key in _multicut.__dict__.keys():
    
    if key not in ["__spec__","__doc__"]:
        try:
                            
            value.__module__='nifty.graph.optimization.multicut'
        except Exception as e:
            continue
        __all__.append(key)



def ilpSettings(relativeGap=0.0, absoluteGap=0.0, memLimit=-1.0):
    """factory function to create :class:`IlpBackendSettigns` .
    
    factory function to create :class:`IlpBackendSettigns` .
    This settings might be consumed by ILP solvers
    as CPLEX GUROBI and GLPK.
        
    Args:
        relativeGap  (float): relative optimality gap (default: {0.0})
        absoluteGap  (float): absolute optimality gap (default: {0.0})
        memLimit (float): memory limit in mega-bites 
            a value smaller as zero indicates no limit  (default: {-1.0})
    
    Returns:
        :class:`IlpBackendSettings`: ilpSettings
    """
    s = IlpBackendSettings()
    s.relativeGap = float(relativeGap)
    s.absoluteGap = float(absoluteGap)
    s.memLimit = float(memLimit)

    return s


def __extendMulticutObj(objectiveCls, objectiveName, graphCls):


    def getCls(prefix, postfix):
        return _multicut.__dict__[prefix+postfix]

    def getSettingsCls(baseName):
        S =  getCls("__"+baseName + "Settings" ,objectiveName)
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

    def factoryClsName(baseName):
        return baseName + "Factory" + objectiveName


    O = objectiveCls

    def verboseVisitor(visitNth=1,timeLimitSolver=float('inf'), 
                       timeLimitTotal=float('inf')):
        V = getMcCls("VerboseVisitor")
        return V(int(visitNth),float(timeLimitSolver),float(timeLimitTotal))
    O.verboseVisitor = staticmethod(verboseVisitor)


    def loggingVisitor(visitNth=1,verbose=True,timeLimitSolver=float('inf'),
                      timeLimitTotal=float('inf')):
        V = getMcCls("LoggingVisitor")
        return V(visitNth=int(visitNth),
                verbose=bool(verbose),
                timeLimitSolver=float(timeLimitSolver),
                timeLimitTotal=float(timeLimitTotal))
    O.loggingVisitor = staticmethod(loggingVisitor)



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
    O.greedyAdditiveFactory.__doc__ = """ create an instance of :class:`%s`

        Find approximate solutions via
        agglomerative clustering as in :cite:`beier_15_funsion`.

    Warning:
        This solver should be used to
        warm start other solvers with.
        This solver is very fast but
        yields rather suboptimal results.

    Args:
        weightStopCond (float): stop clustering when the highest
            weight in cluster-graph is lower as this value (default: {0.0})
        nodeNumStopCond: stop clustering when a cluster-graph
            reached a certain number of nodes.
            Numbers smaller 1 are interpreted as fraction 
            of the graphs number of nodes.
            If nodeNumStopCond is smaller 0 this
            stopping condition is ignored  (default: {-1})
    Returns:    
        %s : multicut factory
    """%(factoryClsName("MulticutGreedyAdditive"),factoryClsName("MulticutGreedyAdditive"))


    def chainedSolversFactory(multicutFactories):
        s,F = getSettingsAndFactoryCls("ChainedSolvers")
        s.multicutFactories = multicutFactories
        return F(s)
    O.chainedSolversFactory = staticmethod(chainedSolversFactory)

    O.chainedSolversFactory.__doc__ = """ create an instance of :class:`%s`

        Chain multiple solvers
        such that each successor is warm-started with 
        its predecessor solver.

    Warning:
        The solvers should be able to be warm started.

    Args:
        weightStopCond (float): stop clustering when the highest
            weight in cluster-graph is lower as this value (default: {0.0})
        nodeNumStopCond: stop clustering when a cluster-graph
            reached a certain number of nodes.
            Numbers smaller 1 are interpreted as fraction 
            of the graphs number of nodes.
            If nodeNumStopCond is smaller 0 this
            stopping condition is ignored  (default: {-1})
    Returns:    
        %s : multicut factory
    """%(factoryClsName("ChainedSolvers"),factoryClsName("ChainedSolvers"))



    def cgcFactory(doCutPhase=True, doGlueAndCutPhase=True, mincutFactory=None,
            multicutFactory=None,
            doBetterCutPhase=False, nodeNumStopCond=0.1, sizeRegularizer=1.0):
        if mincutFactory is None:
            if Configuration.WITH_QPBO:
                mincutFactory = graphCls.MincutObjective.mincutQpboFactory(improve=True)
            else:
                raise RuntimeError("default mincutFactory needs to be compiled WITH_QPBO")

        if Configuration.WITH_QPBO:
            s,F = getSettingsAndFactoryCls("Cgc")
            s.doCutPhase = bool(doCutPhase)
            s.doGlueAndCutPhase = bool(doGlueAndCutPhase)
            s.mincutFactory = mincutFactory
            if multicutFactory is not None:
                s.multicutFactory = multicutFactory
            s.doBetterCutPhase = bool(doBetterCutPhase)
            s.nodeNumStopCond = float(nodeNumStopCond)
            s.sizeRegularizer = float(sizeRegularizer)
            return F(s)
        else:
            raise RuntimeError("cgc need nifty to be compiled WITH_QPBO")
    O.cgcFactory = staticmethod(cgcFactory)
    O.cgcFactory.__module__ = "nifty.graph.optimization.multicut"
    O.cgcFactory.__doc__ = """ create an instance of :class:`%s`

        Cut glue and cut as described in :cite:`beier_14_cut`.

    Warning:
        This solver should be warm started, otherwise 
        the glue phase is very slow.
        Using :func:`greedyAdditiveFactory` to create 
        a solver for warm starting is suggested.


    Note:
        In contrast to the OpenGM implementation we allow for
        arbitrary solvers to optimize the mincut problem.

    Args:
        doCutPhase: do recursive two coloring (default: {True})
        doGlueAndCutPhase: do re-optimization of all pairs of clusters (default: {True})
        mincutFactory: mincutFactory for creating mincut solvers to solve subproblems (default: {None})
        multicutFactory: multicutFactory for creating multicut solvers to solve subproblems (default: {None})
        doBetterCutPhase: do a cut phase with multicut solvers instead of mincuts  (default: {False})
        nodeNumStopCond: If doBetterCutPhase is True, we use a agglomeration to
            create a set of clusters. Each cluster is then optimized with the solver from
            the multicutFactory. Values between 0 and 1 are interpreted as fraction
            of the total number of nodes in the graph (default: {0.1})
        sizeRegularizer: If doBetterCutPhase is True, we use a agglomeration to
            create a set of clusters.
            If this number is larger as zero, the clusters have about equal size (default: {1.0})
    Returns:    
        %s : multicut factory
    """%(factoryClsName("Cgc"),factoryClsName("Cgc"))


    def defaultMulticutFactory():
        return O.greedyAdditiveFactory()
    O.defaultMulticutFactory = staticmethod(defaultMulticutFactory)
    O.defaultFactory = staticmethod(defaultMulticutFactory)
    O.defaultFactory.__doc__ = """ create a instance of the default multicut solver factory.

        Currently the this function returns the same as
        :func:`greedyAdditiveFactory`

    Returns:    
        %s : multicut factory
    """%(factoryClsName("MulticutGreedyAdditive"))



    def multicutAndresGreedyAdditiveFactory():
        s, F = getSettingsAndFactoryCls("MulticutAndresGreedyAdditive")
        return F(s)
    O.multicutAndresGreedyAdditiveFactory = staticmethod(multicutAndresGreedyAdditiveFactory)
    O.multicutAndresGreedyAdditiveFactory.__doc__ = """ create an instance of :class:`%s`

        Find approximate solutions via
        agglomerative clustering as in :cite:`beier_15_funsion`.

    Note:
        This is just for comparison since it implements the
        same as :func:`greedyAddtiveFactory`.


    Returns:    
        %s : multicut factory
    """%tuple([factoryClsName("MulticutAndresGreedyAdditive")]*2)



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
    O.multicutAndresKernighanLinFactory.__doc__ = """ create an instance of :class:`%s`

        Find approximate solutions via
        agglomerative clustering as in :cite:`TODO`.

    Note:
        This is just for comparison since it implements the
        same as :func:`greedyAddtiveFactory`.

    Args:
        numberOfInnerIterations (int): number of inner iterations (default: {sys.maxsize})
        numberOfOuterIterations (int): number of outer iterations        (default: {100})
        epsilon (float): epsilon   (default: { 1e-6})
        verbose (bool):                (default: {False})
        greedyWarmstart (bool): initialize with greedyAdditive  (default: {True})


    Returns:    
        %s : multicut factory
    """%tuple([factoryClsName("MulticutAndresGreedyAdditive")]*2)


    def multicutDecomposerFactory(submodelFactory=None, fallthroughFactory=None):

        if submodelFactory is None:
           submodelFactory = MulticutObjectiveUndirectedGraph.defaultMulticutFactory()

        if fallthroughFactory is None:
            fallthroughFactory = O.defaultMulticutFactory()


        s,F = getSettingsAndFactoryCls("MulticutDecomposer")
        s.submodelFactory = submodelFactory
        s.fallthroughFactory = fallthroughFactory
        return F(s)

    O.multicutDecomposerFactory = staticmethod(multicutDecomposerFactory)
    O.multicutDecomposerFactory.__doc__ = """ create an instance of :class:`%s`

        This solver tries to decompose the model into
        sub-models  as described in :cite:`alush_2013_simbad`.
        If a model decomposes into components such that there are no
        positive weighted edges between the components one can
        optimize each model separately.

        

    Note:
        Models might not decompose at all.

    Args:
        submodelFactory: multicut factory for solving subproblems 
            if model decomposes (default: {:func:`defaultMulticutFactory()`})
        fallthroughFactory: multicut factory for solving subproblems 
            if model does not decompose (default: {:func:`defaultMulticutFactory()`})

    Returns:
        %s : multicut factory
    """%(factoryClsName("MulticutDecomposer"),factoryClsName("MulticutDecomposer"))



    def multicutIlpFactory(addThreeCyclesConstraints=True,
                            addOnlyViolatedThreeCyclesConstraints=True,
                            ilpSolverSettings=None,
                            ilpSolver = None):
        # default solver:
        if ilpSolver is None and Configuration.WITH_CPLEX:
            ilpSolver = 'cplex'
        if ilpSolver is None and Configuration.WITH_GUROBI:
            ilpSolver = 'gurobi'
        if ilpSolver is None and Configuration.WITH_GLPK:
            ilpSolver = 'glpk'
        if ilpSolver is None:
            raise RuntimeError("multicutIlpFactory needs either "
                               "'WITH_CPLEX', 'WITH_GUROBI'"
                               " or 'WITH_GLPK'  to be enabled")

        if ilpSolver == 'cplex':
            if not Configuration.WITH_CPLEX:
                raise RuntimeError("multicutIlpFactory with ilpSolver=`cplex` need nifty "
                        "to be compiled with WITH_CPLEX")
            s,F = getSettingsAndFactoryCls("MulticutIlpCplex")
        elif ilpSolver == 'gurobi':
            if not Configuration.WITH_GUROBI:
                raise RuntimeError("multicutIlpFactory with ilpSolver=`gurobi` need nifty "
                        "to be compiled with WITH_GUROBI")
            s,F = getSettingsAndFactoryCls("MulticutIlpGurobi")
        elif ilpSolver == 'glpk':
            if not Configuration.WITH_GLPK:
                raise RuntimeError("multicutIlpFactory with ilpSolver=`glpk` need nifty "
                        "to be compiled with WITH_GLPK")
            s,F = getSettingsAndFactoryCls("MulticutIlpGlpk")
        else:
            raise RuntimeError("%s is an unknown ilp solver"%str(ilpSolver))
        s.verbose = int(0)
        s.addThreeCyclesConstraints = bool(addThreeCyclesConstraints)
        s.addOnlyViolatedThreeCyclesConstraints = bool(addOnlyViolatedThreeCyclesConstraints)
        if ilpSolverSettings is None:
            ilpSolverSettings = ilpSettings()
        s.ilpSettings = ilpSolverSettings
        return F(s)

    O.multicutIlpFactory = staticmethod(multicutIlpFactory)
    if Configuration.WITH_CPLEX:
        O.multicutIlpCplexFactory = staticmethod(partial(multicutIlpFactory,ilpSolver='cplex'))
    if Configuration.WITH_GUROBI:
        O.multicutIlpGurobiFactory = staticmethod(partial(multicutIlpFactory,ilpSolver='gurobi'))
    if Configuration.WITH_GLPK:
        O.multicutIlpGlpkFactory = staticmethod(partial(multicutIlpFactory,ilpSolver='glpk'))

    O.multicutIlpFactory.__doc__ = """ create an instance of an ilp multicut solver.

        Find a global optimal solution by a cutting plane ILP solver
        as described in :cite:`Kappes-2011` 
        and :cite:`andres_2011_probabilistic` 
        

    Note:
        This might take very long for large models.

    Args:
        addThreeCyclesConstraints (bool) : 
            explicitly add constraints for cycles
            of length 3 before optimization (default: {True})
        addOnlyViolatedThreeCyclesConstraints (bool) :
            explicitly add all violated constraints for only violated cycles
            of length 3 before optimization (default: {True})
        ilpSolverSettings (:class:`IlpBackendSettings`) :
            Settings of the ilp solver (default : {:func:`ilpSettings`})
        ilpSolver (str) : name of the solver. Must be in
            either "cplex", "gurobi" or "glpk".
            "glpk" is only capable of solving very small models. 
            (default: {"cplex"}).
            
    Returns:
        %s or %s or %s : multicut factory for the corresponding solver

    """%(      
        factoryClsName("MulticutIlpCplex"),
        factoryClsName("MulticutIlpGurobi"),
        factoryClsName("MulticutIlpGlpk"),
    )


    # O.multicutIlpCplexFactory.__doc__ = 
    # O.multicutIlpGurobiFactory
    # O.multicutIlpGlpkFactory



    if Configuration.WITH_LP_MP:
        def multicutMpFactory(
                mcFactory = None,
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
            if Configuration.WITH_CPLEX:
                mcFactory = MulticutObjectiveUndirectedGraph.multicutIlpCplexFactory()
            else:
                mcFactory = MulticutObjectiveUndirectedGraph.defaultMulticutFactory()
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




    def watershedCcProposals(sigma=1.0, numberOfSeeds=0.1):
        s,F = getSettingsAndFactoryCls("WatershedProposalGenerator")
        s.sigma = float(sigma)
        s.numberOfSeeds = float(numberOfSeeds)
        return F(s)
    O.watershedCcProposals = staticmethod(watershedCcProposals)


    def interfaceFlipperCcProposals():
        s,F = getSettingsAndFactoryCls("InterfaceFlipperProposalGenerator")
        return F(s)
    O.interfaceFlipperCcProposals = staticmethod(interfaceFlipperCcProposals)


    def ramdomNodeColorCcProposals(numberOfColors=2):
        s,F = getSettingsAndFactoryCls("RandomNodeColorProposalGenerator")
        s.numberOfColors = int(numberOfColors)
        return F(s)
    O.ramdomNodeColorCcProposals = staticmethod(ramdomNodeColorCcProposals)


    def ccFusionMoveBasedFactory(proposalGenerator=None,
        numberOfThreads=1, numberOfIterations=100,
        stopIfNoImprovement=10, fusionMove=None):


        solverSettings,F = getSettingsAndFactoryCls("CcFusionMoveBased")

        if proposalGenerator is None:
            proposalGenerator = watershedCcProposals()
        if fusionMove is None:
            fusionMove = fusionMoveSettings()



        solverSettings.fusionMoveSettings = fusionMove
        solverSettings.proposalGenerator = proposalGenerator
        solverSettings.numberOfIterations = int(numberOfIterations)
        solverSettings.stopIfNoImprovement = int(stopIfNoImprovement)
        solverSettings.numberOfThreads = int(numberOfThreads)
        factory = F(solverSettings)
        return factory
    O.ccFusionMoveBasedFactory = staticmethod(ccFusionMoveBasedFactory)








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












# def __extendDocstrings():

#     mcDocstring =  Multicut Objective for an %s

#         .. math::

#        (a + b)^2  &=  (a + b)(a + b) \\
#                   &=  a^2 + 2ab + b^2
                  

    

#     # hack docstrings
#     MulticutObjectiveUndirectedGraph.__doc__ = mcDocstring % ("UndirectedGraph")


# __extendDocstrings()
# del(__extendDocstrings)