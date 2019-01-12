from __future__ import absolute_import
from .import _lifted_multicut as __lifted_multicut
from ._lifted_multicut import *
from functools import partial
from ..multicut import ilpSettings
from .. import Configuration
import numpy

__all__ = []
for key in __lifted_multicut.__dict__.keys():
    __all__.append(key)

    try:
        __lifted_multicut.__dict__[key].__module__='nifty.graph.opt.lifted_multicut'
    except:
        pass


class PixelWiseLmcObjective(object):
    def __init__(self, weights, offsets):

        self.weights = weights
        self.offsets = offsets

        if(self.offsets.shape[1] == 2):
            assert weights.shape[2] == offsets.shape[0]
            self.shape = weights.shape[0:2]
            self.n_variables = self.shape[0]*self.shape[1]
            self.ndim = 2
            self._obj = PixelWiseLmcObjective2D(self.weights, self.offsets)
        elif (self.offsets.shape[1] == 3):
            self.shape = weights.shape[0:3]
            self.n_variables = self.shape[0]*self.shape[1]*self.shape[2]
            self.ndim = 3
            assert weights.shape[3] == offsets.shape[0]
            self._obj = PixelWiseLmcObjective3D(self.weights, self.offsets)
        else:
            raise NotImplementedError("PixelWiseLmcObjective is only implemented for 2D and 3D images")

    def optimize(self,factory, labels=None):
        if labels is None:
            labels = numpy.arange(self.n_variables).reshape(self.shape)
        return self._obj.optimize(factory, labels)





    def evaluate(self, labels):
        return self._obj.evaluate(labels)

    def cpp_obj(self):
        return self._obj


def pixelWiseLmcObjective(weights, offsets):
    return PixelWiseLmcObjective(weights, offsets)








def __extendLiftedMulticutObj(objectiveCls, objectiveName):

    def insertLiftedEdgesBfs(self, maxDistance, returnDistance = False):
        if returnDistance :
            return self._insertLiftedEdgesBfsReturnDist(maxDistance)
        else:
            self._insertLiftedEdgesBfs(maxDistance)

    objectiveCls.insertLiftedEdgesBfs = insertLiftedEdgesBfs







    def getCls(prefix, postfix):
        return _lifted_multicut.__dict__[prefix+postfix]

    def getSettingsCls(baseName):
        S =  getCls(baseName + "SettingsType" ,objectiveName)
        return S
    def getLmcCls(baseName):
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

    def verboseVisitor(visitNth=1,
                       timeLimitSolver=float('inf'),
                       timeLimitTotal=float('inf')):
        V = getLmcCls("LiftedMulticutVerboseVisitor")
        return V(visitNth, float(timeLimitSolver), float(timeLimitTotal))
    O.verboseVisitor = staticmethod(verboseVisitor)



    def chainedSolversFactory(factories):
        s,F = getSettingsAndFactoryCls("ChainedSolvers")
        s.factories = factories
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


    def watershedProposalGenerator(sigma=1.0, numberOfSeeds=0.1,
                                   seedingStrategy='SEED_FROM_LIFTED'):
        """factory function for a watershed based proposal generator for fusion move based
        lifted multicuts.

        Args:
            sigma (float, optional): The weights are perturbed by a additive
                Gaussian noise n(0,sigma) (default  0.0)

            numberOfSeeds (float, optional): Number of seed to generate.
                A number smaller as one will be interpreted as a fraction of
                the number of nodes (default 0.1)
            seedingStrategy (str, optional): Can be:
                - 'SEED_FROM_LIFTED' : All negative weighted lifted edges
                    can be used to generate seeds.
                - 'SEED_FROM_LOCAL' : All negative weighted local edges
                    can be used to generate seeds.
                - 'SEED_FROM_BOTH' : Both, lifted and local edges
                can be used for seeding

        Returns:
            TYPE: parameter object used construct a WatershedProposalGenerator

        """
        pGenCls = getLmcCls("WatershedProposalGeneratorFactory")
        pGenSettings = getSettings("WatershedProposalGenerator")

        # map string to enum
        stringToEnum = {
            'SEED_FROM_LIFTED' : pGenSettings.SeedingStrategy.SEED_FROM_LIFTED,
            'SEED_FROM_LOCAL' : pGenSettings.SeedingStrategy.SEED_FROM_LOCAL,
            'SEED_FROM_BOTH' : pGenSettings.SeedingStrategy.SEED_FROM_BOTH,
        }
        try:
            enumVal = stringToEnum[seedingStrategy]
        except:
            raise RuntimeError("unkown seedingStrategy '%s': must be either"\
                               "'SEED_FROM_LIFTED','SEED_FROM_LOCAL' or "\
                               " 'SEED_FROM_BOTH' "%str(seedingStrategy))

        pGenSettings.sigma = float(sigma)
        pGenSettings.numberOfSeeds = float(numberOfSeeds)
        pGenSettings.seedingStrategy = enumVal

        return pGenCls(pGenSettings)

    O.watershedProposalGenerator = staticmethod(watershedProposalGenerator)


    def fusionMoveBasedFactory(proposalGenerator=None, numberOfThreads=1,
        numberOfIterations=1000, stopIfNoImprovement=100):
        """factory function for a fusion move based lifted
            multicut solver

        Args:
            proposalGenerator (None, optional): Proposal generator (default watershedProposalGenerator)
            numberOfThreads (int, optional):                (default 1)
            numberOfIterations (int, optional): Maximum number of iterations(default 1000)
            stopIfNoImprovement (int, optional): Stop after n iterations without improvement (default 100)

        Returns:
            TYPE: Description
        """
        if proposalGenerator is None:
            proposalGenerator = watershedProposalGenerator()
        s,F = getSettingsAndFactoryCls("FusionMoveBased")
        s.proposalGenerator = proposalGenerator
        s.numberOfThreads = int(numberOfThreads)
        s.numberOfIterations = int(numberOfIterations)
        s.stopIfNoImprovement = int(stopIfNoImprovement)
        return F(s)

    O.fusionMoveBasedFactory = staticmethod(fusionMoveBasedFactory)


    def liftedMulticutGreedyAdditiveFactory(weightStopCond=0.0, nodeNumStopCond=-1.0):
        s,F = getSettingsAndFactoryCls("LiftedMulticutGreedyAdditive")
        s.weightStopCond = float(weightStopCond)
        s.nodeNumStopCond = float(nodeNumStopCond)
        return F(s)
    O.liftedMulticutGreedyAdditiveFactory = staticmethod(liftedMulticutGreedyAdditiveFactory)


    def liftedMulticutKernighanLinFactory(numberOfOuterIterations=1000000,
                                          numberOfInnerIterations=100,
                                          epsilon=1e-7):
        s,F = getSettingsAndFactoryCls("LiftedMulticutKernighanLin")
        s.numberOfOuterIterations = int(numberOfOuterIterations)
        s.numberOfInnerIterations = int(numberOfInnerIterations)
        s.epsilon = float(epsilon)
        return F(s)
    O.liftedMulticutKernighanLinFactory = staticmethod(liftedMulticutKernighanLinFactory)


    def liftedMulticutAndresKernighanLinFactory(numberOfOuterIterations=1000000,
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


    if Configuration.WITH_LP_MP:
        def liftedMulticutMpFactory(
            lmcFactory = None,
            greedyWarmstart = False,
            tightenSlope = 0.05,
            tightenMinDualImprovementInterval = 0,
            tightenMinDualImprovement = 0.,
            tightenConstraintsPercentage = 0.1,
            tightenConstraintsMax = 0,
            tightenInterval = 10,
            tightenIteration = 100,
            tightenReparametrization = "anisotropic",
            roundingReparametrization = "anisotropic",
            standardReparametrization = "anisotropic",
            tighten = True,
            minDualImprovementInterval = 0,
            minDualImprovement = 0.,
            lowerBoundComputationInterval = 1,
            primalComputationInterval = 5,
            timeout = 0,
            maxIter = 1000,
            numThreads = 1
            ):

            s, F = getSettingsAndFactoryCls("LiftedMulticutMp")
            if lmcFactory is None:
                lmcFactory = LiftedMulticutObjectiveUndirectedGraph.liftedMulticutKernighanLinFactory()

            s.lmcFactory      = lmcFactory
            s.greedyWarmstart = greedyWarmstart
            return F(s)
        O.liftedMulticutMpFactory = staticmethod(liftedMulticutMpFactory)


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


__extendLiftedMulticutObj(LiftedMulticutObjectiveUndirectedGraph,
    "LiftedMulticutObjectiveUndirectedGraph")

__extendLiftedMulticutObj(LiftedMulticutObjectiveUndirectedGridGraph2DSimpleNh,
    "LiftedMulticutObjectiveUndirectedGridGraph2DSimpleNh")

__extendLiftedMulticutObj(LiftedMulticutObjectiveUndirectedGridGraph3DSimpleNh,
    "LiftedMulticutObjectiveUndirectedGridGraph2DSimpleNh")

del __extendLiftedMulticutObj
