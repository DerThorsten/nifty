from __future__ import absolute_import

from . import _mincut as __mincut
from ._mincut import *
from .... import Configuration

from functools import partial

__all__ = []
for key in __mincut.__dict__.keys():
    __all__.append(key)
    try:
        __mincut.__dict__[key].__module__='nifty.graph.opt.mincut'
    except:
        pass





def __extendMincutObj(objectiveCls, objectiveName):



    def getCls(prefix, postfix):
        return _mincut.__dict__[prefix+postfix]

    def getSettingsCls(baseName):
        S =  getCls(baseName + "SettingsType" ,objectiveName)
        return S
    def getSolverCls(baseName):
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

    def mincutVerboseVisitor(visitNth=1,timeLimit=0):
        V = getSolverCls("VerboseVisitor")
        return V(visitNth,timeLimit)
    #O.mincutVerboseVisitor = staticmethod(mincutVerboseVisitor)
    O.verboseVisitor = staticmethod(mincutVerboseVisitor)



    def watershedProposalGenerator(sigma=1.0, numberOfSeeds=0.1,seedingStrategie='SEED_FROM_NEGATIVE'):
        """factory function for a watershed based proposal generator for mincutCcFusionMoveBased
        
        Args:
            sigma (float, optional): The weights are perturbed by a additive 
                Gaussian noise n(0,sigma) (default  0.0)

            numberOfSeeds (float, optional): Number of seed to generate. 
                A number smaller as one will be interpreted as a fraction of 
                the number of nodes (default 0.1)
            seedingStrategie (str, optional): Can be:
                - 'SEED_FROM_NEGATIVE' : All negative weighted  edges
                    can be used to generate seeds. 
                - 'SEED_FROM_ALL' : All edges
                    can be used to generate seeds. 
        
        Returns:
            TYPE: parameter object used construct a WatershedProposalGenerator
        
        """
        pGenCls = getSolverCls("WatershedProposalGeneratorFactory")
        pGenSettings = getSettings("WatershedProposalGenerator")

        # map string to enum
        stringToEnum = {
            'SEED_FROM_NEGATIVE' : pGenSettings.SeedingStrategie.SEED_FROM_NEGATIVE,
            'SEED_FROM_ALL' :      pGenSettings.SeedingStrategie.SEED_FROM_ALL,
        }
        try:
            enumVal = stringToEnum[seedingStrategie]
        except:
            raise RuntimeError("unkown seedingStrategie '%s': must be either"\
                               "'SEED_FROM_NEGATIVE' or 'SEED_FROM_ALL'"%str(seedingStrategie))

        pGenSettings.sigma = float(sigma)
        pGenSettings.numberOfSeeds = float(numberOfSeeds)
        pGenSettings.seedingStrategie = enumVal

        return pGenCls(pGenSettings)

    O.watershedProposalGenerator = staticmethod(watershedProposalGenerator)





    def mincutQpboFactory(improve=True):
        if Configuration.WITH_QPBO:
            s,F = getSettingsAndFactoryCls("MincutQpbo")
            s.improve = bool(improve)
            return F(s)
        else:
            raise RuntimeError("mincutQpbo need nifty to be compiled WITH_QPBO")
    O.mincutQpboFactory = staticmethod(mincutQpboFactory)



    def greedyAdditiveFactory( weightStopCond=0.0, nodeNumStopCond=-1.0, improve=True):
        if Configuration.WITH_QPBO:
            s,F = getSettingsAndFactoryCls("MincutGreedyAdditive")
            s.weightStopCond = float(weightStopCond)
            s.nodeNumStopCond = float(nodeNumStopCond)
            s.improve = bool(improve)
            return F(s)
        else:
            raise RuntimeError("greedyAdditiveFactory need nifty to be compiled WITH_QPBO")
    O.greedyAdditiveFactory = staticmethod(greedyAdditiveFactory)



    def mincutCcFusionMoveSettings(mincutFactory=None):
        if mincutFactory is None:
            if Configuration.WITH_QPBO:
                mincutFactory = MincutObjectiveUndirectedGraph.mincutQpboFactory()
            else:
                raise RuntimeError("default mincutFactory needs mincutQpbo, which need nifty to be compiled WITH_QPBO")

        s = getSettings("MincutCcFusionMove")
        s.mincutFactory = mincutFactory
        return s


    def mincutCcFusionMoveBasedFactory(proposalGenerator=None, numberOfThreads=1,
        numberOfIterations=1000, stopIfNoImprovement=100,
        fusionMoveSettings=None):
        """factory function for a  cc-fusion move based mincut solver
        
        Args:
            proposalGenerator (None, optional): Proposal generator (default watershedProposalGenerator)
            numberOfThreads (int, optional):                (default 1)
            numberOfIterations (int, optional): Maximum number of iterations(default 1000)
            stopIfNoImprovement (int, optional): Stop after n iterations without improvement (default 100)
            fusionMoveSettings (FusionMoveSettings, optional) : The settings of the underlaying mincutCcFusionMove
        Returns:
            TYPE: MincutCcFusionMoveBasedFactory
        """
        if proposalGenerator is None:
            proposalGenerator = watershedProposalGenerator()

        if fusionMoveSettings is None:
            fusionMoveSettings = mincutCcFusionMoveSettings()

        s,F = getSettingsAndFactoryCls("MincutCcFusionMoveBased")
        s.proposalGenerator = proposalGenerator
        s.numberOfThreads = int(numberOfThreads)
        s.numberOfIterations = int(numberOfIterations)
        s.stopIfNoImprovement = int(stopIfNoImprovement)
        s.fusionMoveSettings = fusionMoveSettings
        return F(s)

    O.mincutCcFusionMoveBasedFactory = staticmethod(mincutCcFusionMoveBasedFactory)













__extendMincutObj(MincutObjectiveUndirectedGraph,
    "MincutObjectiveUndirectedGraph")
__extendMincutObj(MincutObjectiveEdgeContractionGraphUndirectedGraph,
    "MincutObjectiveEdgeContractionGraphUndirectedGraph")
del __extendMincutObj
