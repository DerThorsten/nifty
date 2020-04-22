# export CC=x86_64-conda_cos6-linux-gnu-gcc
# export CXX=x86_64-conda_cos6-linux-gnu-g++
export PY_INSTALL_DIR="${PREFIX}/lib/python3.7/site-packages"

##
## START THE BUILD
##

mkdir -p build
cd build

CXXFLAGS="${CXXFLAGS} -I${PREFIX}/include -std=c++17"

##
## Configure
##
cmake .. \
        -DCMAKE_BUILD_TYPE=RELEASE \
        -DCMAKE_PREFIX_PATH="${PREFIX}" \
        -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
        -DPYTHON_MODULE_INSTALL_DIR="${PY_INSTALL_DIR}" \
\
        -DCMAKE_CXX_FLAGS="${CXXFLAGS}" \
\
        -DWITH_HDF5=ON \
        -DWITH_Z5=ON \
        -DWITH_ZLIB=ON \
        -DWITH_BLOSC=ON \
        -DWITH_BZIP2=ON \
        -DWITH_BOOST_FS=OFF \
\
        -DBUILD_NIFTY_PYTHON=ON \
        -DPYTHON_EXECUTABLE="${PYTHON}" \
##

##
## Compile
##
make -j ${CPU_COUNT}
make install
