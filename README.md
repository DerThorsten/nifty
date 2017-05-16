Nifty
========

[![Build Status](https://travis-ci.org/DerThorsten/nifty.svg?branch=master)](https://travis-ci.org/DerThorsten/nifty)

A nifty library for graph based image segmentation.

Almost all features of nifty exist to implement algorithms
for 2D and 3D image segmentation.
To be more precise, nifty was developed to implement
and prototype algorithms for segmentation of neuro data.


Important:
=========
To use nifty one needs to checkout some submodules via:

    git submodule init
    git submodule update

If WITH_MP_LP is active, one needs:

    git submodule update --init --recursive



Features (Highlights):
==================


* Multicut:
    * Multicut-Ilp (Kappes et al. 2011)
        * Multicut-Ilp-Cplex
        * Multicut-Ilp-Gurobi
        * Multicut-Ilp-Glpk
    * Cut Glue & Cut (Beier et al 2014)
        * Cut Glue & Cut - QPBO 
    * Greedy Additive Clustering /  Energy based Hierarchical Clustering (Beier et al. 2015)
    * Fusion Moves for Correlation clustering (Beier et al. 2015)
        * Perturbed Random Seed Watershed Proposal Generator
        * Perturbed Greedy Additive Clustering Proposal Generator
* Lifted Multicut: (Andres et al. 2015, Keuper et al 2015)
    * Kernighan-Lin Algorithm with Joins (Keuper et al 2015)
    * Greedy Additive Clustering (Keuper et al 2015)
    * Lifted-Multicut-Ilp (does not scale to meaningful problems, just for verification)
        * Lifted-Multicut-Ilp-Cplex
        * Lifted-Multicut-Ilp-Gurobi
        * Lifted-Multicut-Ilp-Glpk
    * Fusion Moves for Lifted Multicuts (Beier et al. 2016)
        * Perturbed Random Seed Watershed Proposal Generator
        * Perturbed Greedy Additive Clustering Proposal Generator

* Agglomerative Clustering
    * Easy to extend / Custom cluster policies
    * UCM Transform
* CGP 2D (Cartesian Grid Partitioning)
* Many Data Structures:
    * Union Find Data Structure
    * Histogram

* Coming Eventually =):
    * MinStCut:
    * MultiwayCut:
    * ModifiedMultiwayCut:
    * LiftedModifiedMultiwayCut:



C++ vs Python:
==============
The Python API is at present the easiest to use. The C++ API is mostly for power users.
We recommend to use library from Python.
Almost any class / function in the Python API is calling into C++ via pybind11.

Install:
========

The easiest way to install this version is via conda (right now only available with gurobi, cplex version coming soon):

```
$ GUROBI_ROOT_DIR=/path/to/gurobi conda install -c cpape nifty-with-gurobi
```



Troubleshooting:
=================

TODO

Changelog:
=================

TODO
