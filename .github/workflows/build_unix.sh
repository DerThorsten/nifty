#!/bin/bash

export PY_BIN="$CONDA_PREFIX/bin/python"
cmake . \
    -DWITHIN_TRAVIS=ON \
    -DWITH_QPBO=OFF \
    -DWITH_HDF5=ON \
    -DWITH_Z5=ON \
    -DWITH_ZLIB=ON \
    -DWITH_BLOSC=ON \
    -DWITH_GLPK=OFF \
    -DWITH_CPLEX=OFF \
    -DWITH_GUROBI=OFF \
    -DBUILD_CPP_TEST=OFF \
    -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
    -DPython_NumPy_INCLUDE_DIRS=$(python -c "import numpy; print(numpy.get_include())") \
    -DPYTHON_EXECUTABLE="$PY_BIN" \
    -DCMAKE_CXX_FLAGS="-std=c++17" \
    -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
    -DBUILD_NIFTY_PYTHON=ON
make -j 4
make install
