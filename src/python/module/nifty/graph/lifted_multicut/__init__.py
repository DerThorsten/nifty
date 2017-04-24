from __future__ import absolute_import
from ._lifted_multicut import *
from functools import partial
from ..multicut import ilpSettings
__all__ = []
for key in _lifted_multicut.__dict__.keys():
    __all__.append(key)




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
        S =  getCls(baseName + "Settings" ,objectiveName)
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


    O = objectiveCls

    def verboseVisitor(visitNth=1):
        V = getLmcCls("LiftedMulticutVerboseVisitor")
        return V(visitNth)
    O.verboseVisitor = staticmethod(verboseVisitor)



    def watershedProposalGenerator(sigma=1.0, numberOfSeeds=0.1,seedingStrategy='SEED_FROM_LIFTED'):
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
            'SEED_FROM_LIFTED' : pGenSettings.SeedingStrategie.SEED_FROM_LIFTED,
            'SEED_FROM_LOCAL' : pGenSettings.SeedingStrategie.SEED_FROM_LOCAL,
            'SEED_FROM_BOTH' : pGenSettings.SeedingStrategie.SEED_FROM_BOTH,
        }
        try:
            enumVal = stringToEnum[seedingStrategy]
        except:
            raise RuntimeError("unkown seedingStrategie '%s': must be either"\
                               "'SEED_FROM_LIFTED','SEED_FROM_LOCAL' or "\
                               " 'SEED_FROM_BOTH' "%str(seedingStrategy))

        pGenSettings.sigma = float(sigma)
        pGenSettings.numberOfSeeds = float(numberOfSeeds)
        pGenSettings.seedingStrategie = enumVal

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


__extendLiftedMulticutObj(LiftedMulticutObjectiveUndirectedGraph,
    "LiftedMulticutObjectiveUndirectedGraph")
del __extendLiftedMulticutObj
