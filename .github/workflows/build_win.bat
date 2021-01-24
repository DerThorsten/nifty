cmake . -G "NMake Makefiles" ^
    -DWITH_QPBO=OFF ^
    -DWITH_HDF5=OFF ^
    -DWITH_Z5=OFF ^
    -DWITH_GLPK=OFF ^
    -DWITH_CPLEX=OFF ^
    -DWITH_GUROBI=OFF ^
    -DBUILD_CPP_TEST=OFF ^
    -DCMAKE_PREFIX_PATH="%CONDA_PREFIX%" ^
    -DCMAKE_INSTALL_PREFIX="%CONDA_PREFIX%" ^
    -DBUILD_NIFTY_PYTHON=ON

REM FIXME z5 tests still fail on windows 
REM    -DWITH_Z5=ON ^
REM    -DWITH_ZLIB=ON ^
REM    -DWITH_BLOSC=ON ^

cmake --build . --target INSTALL --config Release -j 4
