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


print("WITH_QPBO", nify.Configuration.WITH_QPBO)
print("WITH_CPLEX", nify.Configuration.WITH_CPLEX)
print("WITH_GURPBI", nify.Configuration.WITH_GURPBI)
print("WITH_GLPK", nify.Configuration.WITH_GLPK)
print("WITH_HDF5", nify.Configuration.WITH_HDF5)
print("WITH_LP_MP", nify.Configuration.WITH_LP_MP)