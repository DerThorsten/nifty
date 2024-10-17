##
## START THE BUILD
##

CXXFLAGS="${CXXFLAGS} -I${PREFIX}/include -std=c++17"

##
## Configure
##
cmake . \
        -DCMAKE_BUILD_TYPE=RELEASE \
        -DCMAKE_PREFIX_PATH="${PREFIX}" \
        -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
\
        -DCMAKE_CXX_FLAGS="${CXXFLAGS}" \
\
        -DWITH_HDF5=ON \
        -DWITH_Z5=ON \
        -DWITH_ZLIB=ON \
        -DWITH_BLOSC=ON \
        -DWITH_BZIP2=OFF \
        -DWITH_BOOST_FS=OFF \
\
        -DBUILD_NIFTY_PYTHON=ON \
        -DPython_EXECUTABLE="${PYTHON}" \
##

##
## Compile
##
make -j ${CPU_COUNT}
make install
