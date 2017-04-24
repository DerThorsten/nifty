##
## This build script is parameterized by the following external environment variables:
## - WITH_CPLEX
##    - Build Nifty with CPLEX enabled
##
## - PREFIX, PYTHON, CPU_COUNT, etc. (as defined by conda-build)

# Convert '0' to empty (all code below treats non-empty as True)
if [[ "$WITH_CPLEX" == "0" ]]; then
    WITH_CPLEX=""
fi

# Platform-specific dylib extension
if [ $(uname) == "Darwin" ]; then
    export DYLIB="dylib"
else
    export DYLIB="so"
fi

# Pre-define special flags, paths, etc. if we're building with CPLEX support.
if [[ "$WITH_CPLEX" == "" ]]; then
    CPLEX_ARGS=""
    LINKER_FLAGS=""
else
    if [ $(echo $PREFIX | grep -q envs)$? -eq 0 ]; then
        ROOT_ENV_PREFIX="${PREFIX}/../.."
    else
        ROOT_ENV_PREFIX="${PREFIX}"
    fi
    CPLEX_LOCATION_CACHE_FILE="${ROOT_ENV_PREFIX}/share/cplex-root-dir.path"
    
    if [[ "$CPLEX_ROOT_DIR" == "<UNDEFINED>" || "$CPLEX_ROOT_DIR" == "" ]]; then
        # Look for CPLEX_ROOT_DIR in the cplex-shared cache file.
        CPLEX_ROOT_DIR=`cat ${CPLEX_LOCATION_CACHE_FILE} 2> /dev/null` \
        || CPLEX_ROOT_DIR="<UNDEFINED>"
    fi
    
    if [ "$CPLEX_ROOT_DIR" == "<UNDEFINED>" ]; then
        set +x
        echo "******************************************"
        echo "* You must define CPLEX_ROOT_DIR in your *"
        echo "* environment before building nifty.     *"
        echo "******************************************"
        exit 1
    fi

    CPLEX_BIN_DIR=`echo $CPLEX_ROOT_DIR/cplex/bin/x86-64*`
    CPLEX_LIB_DIR=`echo $CPLEX_ROOT_DIR/cplex/lib/x86-64*/static_pic`
    CONCERT_LIB_DIR=`echo $CPLEX_ROOT_DIR/concert/lib/x86-64*/static_pic`
            
    #LINKER_FLAGS="-L${PREFIX}/lib -L${CPLEX_LIB_DIR} -L${CONCERT_LIB_DIR}"
    #if [ `uname` != "Darwin" ]; then
    #    LINKER_FLAGS="-Wl,-rpath-link,${PREFIX}/lib ${LINKER_FLAGS}"
    #fi

    CPLEX_LIBRARY=${CPLEX_LIB_DIR}/libcplex.${DYLIB}
    CPLEX_ILOCPLEX_LIBRARY=${CPLEX_LIB_DIR}/libilocplex.${DYLIB}
    CPLEX_CONCERT_LIBRARY=${CONCERT_LIB_DIR}/libconcert.${DYLIB}
    
    set +e
    (
        set -e
        # Verify the existence of the cplex dylibs.
        ls ${CPLEX_LIBRARY}
        ls ${CPLEX_ILOCPLEX_LIBRARY}
        ls ${CPLEX_CONCERT_LIBRARY}
    )
    if [ $? -ne 0 ]; then
        set +x
        echo "************************************************"
        echo "* Your CPLEX installation does not include     *" 
        echo "* the necessary shared libraries.              *"
        echo "*                                              *"
        echo "* Please install the 'cplex-shared' package:   *"
        echo "*                                              *"
        echo "*     $ conda install cplex-shared             *"
        echo "*                                              *"
        echo "* (You only need to do this once per machine.) *"
        echo "************************************************"
        exit 1
    fi
    set -e

    echo "Building with CPLEX from: ${CPLEX_ROOT_DIR}"
    
    CPLEX_ARGS="-DWITH_CPLEX=ON -DCPLEX_ROOT_DIR=${CPLEX_ROOT_DIR}"
    
    # For some reason, CMake can't find these cache variables on even though we give it CPLEX_ROOT_DIR
    # So here we provide the library paths explicitly
    CPLEX_ARGS="${CPLEX_ARGS} -DCPLEX_LIBRARY=${CPLEX_LIBRARY}"
    CPLEX_ARGS="${CPLEX_ARGS} -DCPLEX_ILOCPLEX_LIBRARY=${CPLEX_ILOCPLEX_LIBRARY}"
    CPLEX_ARGS="${CPLEX_ARGS} -DCPLEX_CONCERT_LIBRARY=${CPLEX_CONCERT_LIBRARY}"
    CPLEX_ARGS="${CPLEX_ARGS} -DCPLEX_BIN_DIR=${CPLEX_CONCERT_LIBRARY}"
fi

if [[ "$WITH_GUROBI" == "" ]]; then
    GUROBI_ARGS=""
    LINKER_FLAGS=""
else
    GUROBI_ARGS=""
    GUROBI_ARGS="${GUROBI_ARGS} -DWITH_GUROBI=ON"
    GUROBI_ARGS="${GUROBI_ARGS} -DGUROBI_ROOT_DIR=${GUROBI_ROOT_DIR}"
    GUROBI_ARGS="${GUROBI_ARGS} -DGUROBI_LIBRARY=$(ls ${GUROBI_ROOT_DIR}/lib/libgurobi*.so)"
    GUROBI_ARGS="${GUROBI_ARGS} -DGUROBI_INCLUDE_DIR=${GUROBI_ROOT_DIR}/include"
    
    if [ $(uname) == "Darwin" ]; then    
	    # Note: For Mac, the nice Gurobi people provide two versions of the gurobi library,
	    #       depending on which version of the C++ std library you need to use:
	    #       - For libstdc++ (from the GNU people), use libgurobi_stdc++.a
	    #       - For libc++    (from the clang people), use libgurobi_c++.a
	    #       We use gcc (even on Mac), so we use the libstdc++ version.
	    GUROBI_ARGS="${GUROBI_ARGS} -DGUROBI_CPP_LIBRARY=${GUROBI_ROOT_DIR}/lib/libgurobi_stdc++.a"    
    else
        # Only one choice on Linux. It works with libstdc++ (from the GNU people).
        # (The naming convention isn't consistent with the name on Mac, but that's okay.)
        GUROBI_ARGS="${GUROBI_ARGS} -DGUROBI_CPP_LIBRARY=${GUROBI_ROOT_DIR}/lib/libgurobi_c++.a"    
    fi
fi

##
## START THE BUILD
##

mkdir build
cd build

CXXFLAGS="${CXXFLAGS} -I${PREFIX}/include"
LDFLAGS="${LDFLAGS} -Wl,-rpath,${PREFIX}/lib -L${PREFIX}/lib"

##
## Configure
##
cmake .. \
        -DCMAKE_C_COMPILER=${PREFIX}/bin/gcc \
        -DCMAKE_CXX_COMPILER=${PREFIX}/bin/g++ \
        -DCMAKE_BUILD_TYPE=RELEASE \
        -DCMAKE_OSX_DEPLOYMENT_TARGET=10.7\
        -DCMAKE_INSTALL_PREFIX=${PREFIX} \
        -DCMAKE_PREFIX_PATH=${PREFIX} \
\
        -DCMAKE_SHARED_LINKER_FLAGS="${LDFLAGS}" \
        -DCMAKE_EXE_LINKER_FLAGS="${LDFLAGS}" \
        -DCMAKE_CXX_FLAGS="${CXXFLAGS}" \
        -DCMAKE_CXX_FLAGS_RELEASE="${CXXFLAGS}" \
        -DCMAKE_CXX_FLAGS_DEBUG="${CXXFLAGS}" \
\
        -DBOOST_ROOT=${PREFIX} \
        -DWITH_HDF5=ON \
        -DHDF5_INCLUDE_DIR=${PREFIX}/include \
        -DWITH_FASTFILTERS=ON \
        -DFASTFILTERS_INCLUDE_DIR=${PREFIX}/include\
        ${CPLEX_ARGS} \
        ${GUROBI_ARGS} \
\
        -DBUILD_NIFTY_PYTHON=ON \
        -DPYTHON_EXECUTABLE=${PYTHON} \
        -DPYTHON_LIBRARY=${PREFIX}/lib/libpython2.7.${DYLIB} \
        -DPYTHON_INCLUDE_DIR=${PREFIX}/include/python2.7 \
##

##
## Compile
##
make -j${CPU_COUNT}
#make test

##
## Install to prefix
cp -r ${SRC_DIR}/build/python/nifty ${PREFIX}/lib/python2.7/site-packages/


NIFTY_MODULE_SO=${PREFIX}/lib/python2.7/site-packages/nifty/_nifty.so

##
## Rename the python module entirely, and change cplex lib install names.
##
if [[ "$WITH_CPLEX" != "" ]]; then
    (
        if [ `uname` == "Darwin" ]; then
            # Set install names according using @rpath
            install_name_tool -change ${CPLEX_LIB_DIR}/libcplex.dylib     @rpath/libcplex.dylib    ${NIFTY_MODULE_SO}
            install_name_tool -change ${CPLEX_LIB_DIR}/libilocplex.dylib  @rpath/libilocplex.dylib ${NIFTY_MODULE_SO}
            install_name_tool -change ${CONCERT_LIB_DIR}/libconcert.dylib @rpath/libconcert.dylib  ${NIFTY_MODULE_SO}
        fi

        # Rename the nifty package to 'nifty_with_cplex'
        cd "${PREFIX}/lib/python2.7/site-packages/"
        mv nifty nifty_with_cplex
    )
fi

##
## Rename the python module entirely, and change cplex lib install names.
##
if [[ "$WITH_GUROBI" != "" ]]; then
    (
        if [ `uname` == "Darwin" ]; then
            # Set install name according using @rpath
            fullpath=$(ls ${GUROBI_ROOT_DIR}/lib/libgurobi*.so*)
            install_name_tool -change $fullpath @rpath/$(basename $fullpath) ${NIFTY_MODULE_SO} 
        fi

        # Rename the nifty package to 'nifty_with_gurobi'
        cd "${PREFIX}/lib/python2.7/site-packages/"
        mv nifty nifty_with_gurobi
    )
fi
