# TODO how do we set the python version properly?
conda create -q -n dev -c conda-forge python=3.7 cmake  # gcc_linux-64 gxx_linux-64
source activate dev
conda install -c conda-forge xtensor-python boost-cpp scikit-image h5py vigra z5py>=1.5.1 nlohmann_json
        
export ENV_BIN="$CONDA_PREFIX/bin"
# export CC="$ENV_BIN/x86_64-conda_cos6-linux-gnu-cc"
# export CXX="$ENV_BIN/x86_64-conda_cos6-linux-gnu-c++"
        
$ENV_BIN/cmake \
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
    -DBUILD_NIFTY_PYTHON=ON \
    -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
    -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
    -DCMAKE_CXX_FLAGS="-std=c++17"
