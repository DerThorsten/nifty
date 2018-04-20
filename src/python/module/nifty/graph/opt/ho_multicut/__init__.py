""" HoMulticut module of nifty

This module implements higher order multicut
related functionality.


"""



from __future__ import absolute_import
import sys
from functools import partial
#from . import _multicut as __multicut
from ._ho_multicut import *
from .... import Configuration, LogLevel
from ... import (UndirectedGraph,)

from ..multicut import ilpSettings




def __extendMulticutObj(objectiveCls, objectiveName, graphCls):


    def getCls(prefix, postfix):
        return _ho_multicut.__dict__[prefix+postfix]

    def getSettingsCls(baseName):
        S =  getCls("__"+baseName + "SettingsType" ,objectiveName)
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
                       timeLimitTotal=float('inf'), logLevel=LogLevel.WARN):
        V = getMcCls("VerboseVisitor")
        return V(int(visitNth),float(timeLimitSolver),float(timeLimitTotal),logLevel)
    O.verboseVisitor = staticmethod(verboseVisitor)


    def loggingVisitor(visitNth=1,verbose=True,timeLimitSolver=float('inf'),
                      timeLimitTotal=float('inf'), logLevel=LogLevel.WARN):
        V = getMcCls("LoggingVisitor")
        return V(visitNth=int(visitNth),
                verbose=bool(verbose),
                timeLimitSolver=float(timeLimitSolver),
                timeLimitTotal=float(timeLimitTotal),
                logLevel=logLevel)
    O.loggingVisitor = staticmethod(loggingVisitor)



    def fusionMoveSettings(hoMcFactory=None):
        if hoMcFactory is None:
            if Configuration.WITH_CPLEX:
                hoMcFactory = HoMulticutObjectiveUndirectedGraph.hoMulticutIlpCplexFactory()
            else:
                raise RuntimeError("this needs cplex")

        s = getSettings('FusionMove')
        s.hoMcFactory = hoMcFactory
        return s
    O.fusionMoveSettings = staticmethod(fusionMoveSettings)

 


    def hoMulticutIlpFactory(addThreeCyclesConstraints=True,
                             addOnlyViolatedThreeCyclesConstraints=True,
                             ilpSolverSettings=None,
                             ilpSolver=None,
                             integralHo=False,
                             ilp=True,
                             timeLimit=-1.0,
                             maxIterations=-1):
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
            s,F = getSettingsAndFactoryCls("HoMulticutIlpCplex")
        elif ilpSolver == 'gurobi':
            if not Configuration.WITH_GUROBI:
                raise RuntimeError("multicutIlpFactory with ilpSolver=`gurobi` need nifty "
                        "to be compiled with WITH_GUROBI")
            s,F = getSettingsAndFactoryCls("HoMulticutIlpGurobi")
        elif ilpSolver == 'glpk':
            if not Configuration.WITH_GLPK:
                raise RuntimeError("multicutIlpFactory with ilpSolver=`glpk` need nifty "
                        "to be compiled with WITH_GLPK")
            s,F = getSettingsAndFactoryCls("HoMulticutIlpGlpk")
        else:
            raise RuntimeError("%s is an unknown ilp solver"%str(ilpSolver))
        s.addThreeCyclesConstraints = bool(addThreeCyclesConstraints)
        s.addOnlyViolatedThreeCyclesConstraints = bool(addOnlyViolatedThreeCyclesConstraints)
        if ilpSolverSettings is None:
            ilpSolverSettings = ilpSettings()
        s.ilpSettings = ilpSolverSettings
        s.integralHo = bool(integralHo)
        s.ilp = bool(ilp)
        s.timeLimit = float(timeLimit)
        s.maxIterations = int(maxIterations)
        return F(s)

    O.hoMulticutIlpFactory = staticmethod(hoMulticutIlpFactory)
    if Configuration.WITH_CPLEX:
        O.hoMulticutIlpCplexFactory = staticmethod(partial(hoMulticutIlpFactory,ilpSolver='cplex'))
    if Configuration.WITH_GUROBI:
        O.hoMulticutIlpGurobiFactory = staticmethod(partial(hoMulticutIlpFactory,ilpSolver='gurobi'))
    if Configuration.WITH_GLPK:
        O.hoMulticutIlpGlpkFactory = staticmethod(partial(hoMulticutIlpFactory,ilpSolver='glpk'))

    O.hoMulticutIlpFactory.__doc__ = """ create an instance of an ilp ho multicut solver.

        Find a global optimal solution by a cutting plane ILP solver


    Note:
        This might take very long for large models.

    Args:
        addThreeCyclesConstraints (bool) :
            explicitly add constraints for cycles
            of length 3 before opt (default: {True})
        addOnlyViolatedThreeCyclesConstraints (bool) :
            explicitly add all violated constraints for only violated cycles
            of length 3 before opt (default: {True})
        ilpSolverSettings (:class:`IlpBackendSettings`) :
            SettingsType of the ilp solver (default : {:func:`ilpSettings`})
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



    def hoMulticutDualDecompositionFactory(
        numberOfIterations=100, submodelMcFactory=None, 
        stepSize=0.1,crfSolver='graphcut',
        absoluteGap=0.0000001,
        fusionMove=None
        ):


        s,F = getSettingsAndFactoryCls("HoMulticutDualDecomposition")
       

        if submodelMcFactory is None:
            submodelMcFactory = graphCls.MulticutObjective.multicutIlpFactory()

        s.submodelMcFactory = submodelMcFactory
        s.stepSize = float(stepSize)
        s.absoluteGap = float(absoluteGap)
        s.numberOfIterations = int(numberOfIterations)
        if crfSolver == 'graphcut':
            s.crfSolver = s.graphcut
        elif crfSolver == 'qpbo':
            s.crfSolver = s.qpbo

        if fusionMove is None:
            fusionMove = fusionMoveSettings()
        s.fusionMoveSettings = fusionMove
        return F(s)

    O.hoMulticutDualDecompositionFactory = staticmethod(hoMulticutDualDecompositionFactory)




__extendMulticutObj(HoMulticutObjectiveUndirectedGraph,
    "HoMulticutObjectiveUndirectedGraph",UndirectedGraph)

del __extendMulticutObj


