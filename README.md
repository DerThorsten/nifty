Nifty
========

[![Build Status](https://travis-ci.org/DerThorsten/nifty.svg?branch=master)](https://travis-ci.org/DerThorsten/nifty)

A nifty library for Multicuts and general image and graph segmentation.
This library is used for my personal research. I decided to make it open source
such that other researchers can benefit from my code.
Since I use this code for research,the code is not super stable and  there might be a lot of API changes,
bugs, and other issues (I try to avoid this but it is not my priority to have a bullet proof library)

C++ vs Python:
==============
The Python API is at present the easiest to use. The C++ API is mostly for power users.
We recommend to use library from Python.
Almost any class / function in the Python API is calling into C++ via pybind11.



Features:
=========
* Multicuts
    * Multicut-Ilp (Kappes et al. 2011)
        * Multicut-Ilp-Cplex
        * Multicut-Ilp-Gurobi
        * Multicut-Ilp-Glpk
    * Fusion Moves for Correlation clustering (Beier et al. 2015)
    * Greedy Additive Clustering /  Energy based Hierarchical Clustering (Beier et al. 2015)

* Lifted Multicuts: (Andres et al. 2015, Keuper et al 2015)
    * Kernighan-Lin Algorithm with Joins (Keuper et al 2015)
    * Greedy Additive Clustering (Keuper et al 2015)
    * Lifted-Multicut-Ilp (does not scale to meaningful problems, just for verification)
        * Lifted-Multicut-Ilp-Cplex
        * Lifted-Multicut-Ilp-Gurobi
        * Lifted-Multicut-Ilp-Glpk