Github Workflow builds
---------------------------------
[![Build Status](https://github.com/constantinpape/nifty/workflows/build/badge.svg)](https://github.com/constantinpape/nifty/actions)


Nifty
========


A nifty library for 2D and 3D image segmentation,
graph based segmentation an opt.
This library provided building blocks for segmentation
algorithms and complex segmentation pipelines.
The core is implemented in C++ but
the suggested language to use this library from is
python.

Important:
=========
To use nifty one needs to checkout some submodules via:

    git submodule init
    git submodule update

If WITH_MP_LP is active, one needs:

    git submodule update --init --recursive

Documentation:
===============
A very tentative [documentation of the nifty python
module](http://derthorsten.github.io/nifty/docs/python/html/index.html).


Features (Highlights):
==================


* Multicut:
    * Multicut-Ilp (Kappes et al. 2011)
        * Multicut-Ilp-Cplex
        * Multicut-Ilp-Gurobi
        * Multicut-Ilp-Glpk
    * Decomposing Solver (Alush and Goldberger 2012)
    * Cut Glue & Cut (Beier et al 2014)
        * Cut Glue & Cut - QPBO 
    * Greedy Additive Clustering /  Energy based Hierarchical Clustering (Beier et al. 2015)
    * Fusion Moves for Correlation clustering (Beier et al. 2015)
        * Perturbed Random Seed Watershed Proposal Generator
        * Perturbed Greedy Additive Clustering Proposal Generator
    * Kernighan-Lin Algorithm with Joins (Keuper et al 2015)
    * Message Passing for the Minimum Cost Multicut Problem (Swoboda 2016)

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
    * Message Passing for the Minimum Cost Multicut Problem (Swoboda 2016)


* Mincut/Maxcut:
    * QPBO 

* Agglomerative Clustering
    * Easy to extend / Custom cluster policies
    * UCM Transform
* CGP 2D (Cartesian Grid Partitioning)
* Many Data Structures:
    * Union Find Data Structure
    * Histogram

* Coming Eventually:
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


Conda:
======

The easiest way to install this version is via conda (right now it is not available with gurobi or cplex, you will need to build from source for this):
```
conda install -c conda-forge nifty
```

From Source:
============

To build nifty from source, you will need to set up an environment with all necessary dependencies.
We recommend to use conda for this as well and provide the file environment.yaml` for this.
You can set up this environment and build nifty with the following commands: 

```sh
$ conda env create -f environment.yaml
$ conda activate nifty-dev
$ mkdir bld
$ cd bld
$ cmake -DCMAKE_PREFIX_PATH=/path/to/conda/envs/nifty-dev -DWITH_Z5=ON -DWITH_HDF5=ON -DWITH_ZLIB=ON ..
$ make
```

This would build nifty with additional hdf5 and z5 dependencies; and add zlib support for z5.
You can find all build options in [CMakeLists.txt](https://github.com/DerThorsten/nifty/blob/master/CMakeLists.txt#L36-L50);
note that you may need to install additional dependencies.


Troubleshooting:
=================

TODO

Changelog:
=================

TODO
