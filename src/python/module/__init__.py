from nifty import *





def greedyAdditiveProposals(sigma=1.0, weightStopCond=0.0, nodeNumStopCond=-1.0):
    s = graph.multicut.FusionMoveBasedGreedyAdditiveProposalGenSettingsUndirectedGraph()
    s.sigma = float(sigma)
    s.weightStopCond = float(weightStopCond)
    s.nodeNumStopCond = float(nodeNumStopCond)
    return s

def greedyAdditiveFactory(verbose=0):
    s = graph.multicut.MulticutGreedyAdditiveSettingsUndirectedGraph()
    s.verbose = int(verbose)
    factory = graph.multicut.MulticutGreedyAdditiveFactoryUndirectedGraph(s)
    return factory




def multicutIlpCplexFactory(verbose=0, addThreeCyclesConstraints=True,
                            addOnlyViolatedThreeCyclesConstraints=True):
    s = graph.multicut.MulticutIlpCplexSettingsUndirectedGraph()
    s.verbose = int(verbose)
    s.addThreeCyclesConstraints = bool(addThreeCyclesConstraints)
    s.addOnlyViolatedThreeCyclesConstraints = bool(addOnlyViolatedThreeCyclesConstraints)
    factory = graph.multicut.MulticutIlpCplexFactoryUndirectedGraph(s)
    return factory

def multicutIlpGurobiFactory(verbose=0, addThreeCyclesConstraints=True,
                            addOnlyViolatedThreeCyclesConstraints=True):
    s = graph.multicut.MulticutIlpGurobiSettingsUndirectedGraph()
    s.verbose = int(verbose)
    s.addThreeCyclesConstraints = bool(addThreeCyclesConstraints)
    s.addOnlyViolatedThreeCyclesConstraints = bool(addOnlyViolatedThreeCyclesConstraints)
    factory = graph.multicut.MulticutIlpGurobiFactoryUndirectedGraph(s)
    return factory

def multicutIlpFactory( ilpSolver = 'cplex',
                            verbose=0, addThreeCyclesConstraints=True,
                            addOnlyViolatedThreeCyclesConstraints=True):
    
    if ilpSolver == 'cplex':
        f = multicutIlpCplexFactory
    else:
        f = multicutIlpGurobiFactory
    return f(verbose=verbose,addThreeCyclesConstraints=addThreeCyclesConstraints,
            addOnlyViolatedThreeCyclesConstraints=addOnlyViolatedThreeCyclesConstraints)

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
    else:
        assert False
    solverSettings.fusionMoveSettings = fusionMove
    solverSettings.proposalGenSettings = proposalGen
    solverSettings.numberOfIterations = int(numberOfIterations)
    solverSettings.verbose = int(verbose)
    solverSettings.numberOfParallelProposals = int(numberOfParallelProposals)
    solverSettings.fuseN = int(fuseN)
    solverSettings.stopIfNoImprovement = int(stopIfNoImprovement)

    factory = factoryCls(solverSettings)
    return factory
