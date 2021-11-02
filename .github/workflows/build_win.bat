REM https://stackoverflow.com/questions/6832666/lnk2019-when-including-asio-headers-solution-generated-with-cmake
cmake . -G "NMake Makefiles" ^
    -DWITH_QPBO=OFF ^
    -DWITH_HDF5=OFF ^
    -DWITH_GLPK=OFF ^
    -DWITH_CPLEX=OFF ^
    -DWITH_GUROBI=OFF ^
    -DBUILD_CPP_TEST=OFF ^
    -DCMAKE_PREFIX_PATH="%CONDA_PREFIX%" ^
    -DCMAKE_INSTALL_PREFIX="%CONDA_PREFIX%" ^
    -DCMAKE_CXX_FLAGS="/std:c++17 /EHsc" ^
    -DBUILD_NIFTY_PYTHON=ON ^
    -DWITH_HDF5=OFF ^
    -DWITH_Z5=ON ^
    -DWITH_ZLIB=ON ^
    -DWITH_BLOSC=ON

cmake --build . --target INSTALL --config Release -j 4
