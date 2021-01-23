cmake . -G "NMake Makefiles" ^
    -DWITH_QPBO=OFF ^
    -DWITH_HDF5=ON ^
    -DWITH_Z5=ON ^
    -DWITH_ZLIB=ON ^
    -DWITH_BLOSC=ON ^
    -DWITH_GLPK=OFF ^
    -DWITH_CPLEX=OFF ^
    -DWITH_GUROBI=OFF ^
    -DBUILD_CPP_TEST=OFF ^
    -DCMAKE_PREFIX_PATH="%CONDA_PREFIX%" ^
    -DCMAKE_INSTALL_PREFIX="%CONDA_PREFIX%" ^
    -DBUILD_NIFTY_PYTHON=ON
cmake --build . --target INSTALL --config Release
