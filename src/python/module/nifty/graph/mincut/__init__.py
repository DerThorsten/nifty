from __future__ import absolute_import
from ._mincut import *
from ... import Configuration

from functools import partial

__all__ = []
for key in _mincut.__dict__.keys():
    __all__.append(key)






def __extendMincutObj(objectiveCls, objectiveName):



    def getCls(prefix, postfix):
        return _mincut.__dict__[prefix+postfix]

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

    def mincutVerboseVisitor(visitNth=1,timeLimit=0):
        V = getMcCls("MincutVerboseVisitor")
        return V(visitNth,timeLimit)
    O.mincutVerboseVisitor = staticmethod(mincutVerboseVisitor)
    O.verboseVisitor = staticmethod(mincutVerboseVisitor)

    def mincutQpboFactory(improve=True):
        if Configuration.WITH_QPBO:
            s,F = getSettingsAndFactoryCls("MincutQpbo")
            s.improve = bool(improve)
            return F(s)
        else:
            raise RuntimeError("mincutQpbo need nifty to be compiled WITH_QPBO")
    O.mincutQpboFactory = staticmethod(mincutQpboFactory)



__extendMincutObj(MincutObjectiveUndirectedGraph,
    "MincutObjectiveUndirectedGraph")
__extendMincutObj(MincutObjectiveEdgeContractionGraphUndirectedGraph,
    "MincutObjectiveEdgeContractionGraphUndirectedGraph")
del __extendMincutObj
