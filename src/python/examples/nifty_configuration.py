"""
Configuration
====================================

Nifty can be build optional components 
with certain optional dependencies.
This example show how to check
this build configuration.
"""
from __future__ import print_function
import nifty


print("WITH_QPBO", nifty.Configuration.WITH_QPBO)
print("WITH_CPLEX", nifty.Configuration.WITH_CPLEX)
print("WITH_GURPBI", nifty.Configuration.WITH_GURPBI)
print("WITH_GLPK", nifty.Configuration.WITH_GLPK)
print("WITH_HDF5", nifty.Configuration.WITH_HDF5)
print("WITH_LP_MP", nifty.Configuration.WITH_LP_MP)