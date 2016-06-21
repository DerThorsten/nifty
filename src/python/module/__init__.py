from nifty import *





def greedyAdditiveProposals(sigma=1.0, weightStopCond=0.0, nodeNumStopCond=-1.0):
    s = graph.multicut.FusionMoveBasedGreedyAdditiveProposalGenSettingsUndirectedGraph()
    s.sigma = float(sigma)
    s.weightStopCond = float(weightStopCond)
    s.nodeNumStopCond = float(nodeNumStopCond)
    return s

def watershedProposals(sigma=1.0, seedFraction=0.0):
    s = graph.multicut.FusionMoveBasedWatershedProposalGenSettingsUndirectedGraph()
    s.sigma = float(sigma)
    s.seedFraction = float(seedFraction)
    return s

def greedyAdditiveFactory(verbose=0):
    s = graph.multicut.MulticutGreedyAdditiveSettingsUndirectedGraph()
    s.verbose = int(verbose)
    factory = graph.multicut.MulticutGreedyAdditiveFactoryUndirectedGraph(s)
    return factory




def ilpSettings(relativeGap=0.0, absoluteGap=0.0, memLimit=-1.0):
    s = graph.multicut.IlpBackendSettings()
    s.relativeGap = float(relativeGap)
    s.absoluteGap = float(absoluteGap)
    s.memLimit = float(memLimit)

    return s

def multicutIlpCplexFactory(verbose=0, addThreeCyclesConstraints=True,
                            addOnlyViolatedThreeCyclesConstraints=True,
                            relativeGap=0.0, absoluteGap=0.0, memLimit=-1.0
                            ):
    s = graph.multicut.MulticutIlpCplexSettingsUndirectedGraph()
    s.verbose = int(verbose)
    s.addThreeCyclesConstraints = bool(addThreeCyclesConstraints)
    s.addOnlyViolatedThreeCyclesConstraints = bool(addOnlyViolatedThreeCyclesConstraints)
    s.ilpSettings = ilpSettings(relativeGap=relativeGap, absoluteGap=absoluteGap, memLimit=memLimit)
    factory = graph.multicut.MulticutIlpCplexFactoryUndirectedGraph(s)
    return factory

def multicutIlpGurobiFactory(verbose=0, addThreeCyclesConstraints=True,
                            addOnlyViolatedThreeCyclesConstraints=True,
                            relativeGap=0.0, absoluteGap=0.0, memLimit=-1.0):
    s = graph.multicut.MulticutIlpGurobiSettingsUndirectedGraph()
    s.verbose = int(verbose)
    s.addThreeCyclesConstraints = bool(addThreeCyclesConstraints)
    s.addOnlyViolatedThreeCyclesConstraints = bool(addOnlyViolatedThreeCyclesConstraints)
    s.ilpSettings = ilpSettings(relativeGap=relativeGap, absoluteGap=absoluteGap, memLimit=memLimit)
    factory = graph.multicut.MulticutIlpGurobiFactoryUndirectedGraph(s)
    return factory

def multicutIlpFactory( ilpSolver = 'cplex',
                        verbose=0, addThreeCyclesConstraints=True,
                        addOnlyViolatedThreeCyclesConstraints=True,
                        relativeGap=0.0, absoluteGap=0.0, memLimit=-1.0):
    
    if ilpSolver == 'cplex':
        f = multicutIlpCplexFactory
    else:
        f = multicutIlpGurobiFactory
    return f(verbose=verbose,addThreeCyclesConstraints=addThreeCyclesConstraints,
            addOnlyViolatedThreeCyclesConstraints=addOnlyViolatedThreeCyclesConstraints,
            relativeGap=relativeGap, absoluteGap=absoluteGap, memLimit=memLimit)

def fusionMoveSettings(mcFactory=greedyAdditiveFactory()):
    s = graph.multicut.FusionMoveSettingsUndirectedGraph()
    s.mcFactory = mcFactory
    return s

def fusionMoveBasedFactory(numberOfIterations=10,verbose=0,
                           numberOfParallelProposals=10, fuseN=2,
                           stopIfNoImprovement=4,
                           proposalGen=greedyAdditiveProposals(),
                           fusionMove=fusionMoveSettings()):
    solverSettings = None
    if isinstance(proposalGen, graph.multicut.FusionMoveBasedGreedyAdditiveProposalGenSettingsUndirectedGraph):
        solverSettings = graph.multicut.FusionMoveBasedGreedyAdditiveSettingsUndirectedGraph()
        factoryCls = graph.multicut.FusionMoveBasedGreedyAdditiveFactoryUndirectedGraph
    elif isinstance(proposalGen, graph.multicut.FusionMoveBasedWatershedProposalGenSettingsUndirectedGraph):
        solverSettings = graph.multicut.FusionMoveBasedWatershedSettingsUndirectedGraph()
        factoryCls = graph.multicut.FusionMoveBasedWatershedFactoryUndirectedGraph
    else:
        raise TypeError(str(proposalGen)+" is of unknown type")

    solverSettings.fusionMoveSettings = fusionMove
    solverSettings.proposalGenSettings = proposalGen
    solverSettings.numberOfIterations = int(numberOfIterations)
    solverSettings.verbose = int(verbose)
    solverSettings.numberOfParallelProposals = int(numberOfParallelProposals)
    solverSettings.fuseN = int(fuseN)
    solverSettings.stopIfNoImprovement = int(stopIfNoImprovement)

    factory = factoryCls(solverSettings)
    return factory
