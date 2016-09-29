from _lifted_multicut import *
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
