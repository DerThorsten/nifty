cmake_minimum_required(VERSION 3.1)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

project(nifty)

set (NIFTY_VERSION_MAJOR 0)
set (NIFTY_VERSION_MINOR 18)
set (NIFTY_VERSION_PATCH 0)


set (NIFTY_VERSION_SHORT_STR "${NIFTY_VERSION_MAJOR}.${NIFTY_VERSION_MINOR}")
set (NIFTY_VERSION_STR "${NIFTY_VERSION_MAJOR}.${NIFTY_VERSION_MINOR}.${NIFTY_VERSION_PATCH}")


include(CheckCXXCompilerFlag)



option(WITH_GUROBI OFF)
option(WITH_CPLEX OFF)
option(WITH_GLPK OFF)
option(WITH_HDF5 OFF)
option(WITH_FASTFILTERS OFF)
option(WITH_LP_MP OFF)
option(WITH_QPBO ON)

option(BUILD_CPP_TEST_WITH_GLIBCXX_CXX11_ABI OFF)

option(BUILD_PYTHON_TEST OFF)
option(BUILD_CPP_TEST ON)
option(BUILD_NIFTY_PYTHON OFF)
option(BUILD_DOCS ON)
option(BUILD_CPP_EXAMPLES ON)
option(BUILD_PYTHON_DOCS "Build the Python documentation with Sphinx" OFF)

option(REMOVE_SOME_WARNINGS "Remove some annoying warnings" ON)

if(MSVC)
    # Disable autolinking on MSVC.
    add_definitions(-DBOOST_ALL_NO_LIB)
endif()

# from externals..
if(REMOVE_SOME_WARNINGS AND (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
    add_definitions(-Wno-attributes)
    # vigra uses old stuff....
    add_definitions(-Wno-deprecated-declarations)
    if(WITH_CPLEX)
        # cplex
        add_definitions(-Wno-ignored-attributes)
    endif()
    if(WITH_LP_MP)
        add_definitions(-Wno-non-template-friend)
        add_definitions(-Wno-terminate)
        add_definitions(-Wno-write-strings)
    endif()
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_definitions(-fext-numeric-literals)
endif()
#-------------------------------------------------------------------------------------------------------------------
# check for c++ 14 support
#-------------------------------------------------------------------------------------------------------------------

# FIXME c++14 checks are broken!
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    CHECK_CXX_COMPILER_FLAG("-std=c++14" HAS_CPP14_FLAG)
    CHECK_CXX_COMPILER_FLAG("-std=c++11" HAS_CPP11_FLAG)
    if (HAS_CPP14_FLAG)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    elseif (HAS_CPP11_FLAG)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    else()
        message(FATAL_ERROR "Unsupported compiler -- nifty  requires C++11 support!")
    endif()
endif()

# WITH_LP_MP needs special treatment since
# it requires C++ 14 and used OpenMP
if(WITH_LP_MP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    find_package(OpenMP)
    if (OPENMP_FOUND)
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()

endif()


#-------------------------------------------------------------------------------------------------------------------
# Set a default build configuraton if none is specified. 'MinSizeRel' produces the smallest binaries
#-------------------------------------------------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'MinSizeRel' as none was specified.")
    set(CMAKE_BUILD_TYPE MinSizeRel CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
        "MinSizeRel" "RelWithDebInfo")
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)



#-------------------------------------------------------------------------------------------------------------------
# thread libs
#-------------------------------------------------------------------------------------------------------------------
find_package(Threads)


#-------------------------------------------------------------------------------------------------------------------
# rt lib
#-------------------------------------------------------------------------------------------------------------------
if(UNIX AND NOT APPLE)
  find_library(RT rt)
  set(LINK_RT true)
  #message(STATUS "Linking to RT is enabled")
else()
  set(LINK_RT false)
  #message(STATUS "Linking to RT is diabled")
endif()


#-------------------------------------------------------------------------------------------------------------------
# externals
#-------------------------------------------------------------------------------------------------------------------

include_directories( "${CMAKE_CURRENT_SOURCE_DIR}/externals/qpbo")
#include_directories( "${CMAKE_CURRENT_SOURCE_DIR}/externals/marray/include")
include_directories( "${CMAKE_CURRENT_SOURCE_DIR}/externals/pybind11/include")
include_directories( "${CMAKE_CURRENT_SOURCE_DIR}/externals/vigra/include")
include_directories( "${CMAKE_CURRENT_SOURCE_DIR}/externals/graph/include")

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/externals/pybind11")



#-------------------------------------------------------------------------------------------------------------------
# hdf5 libs
#-------------------------------------------------------------------------------------------------------------------

if(WITH_HDF5)
    find_package(HDF5)
    include_directories(${HDF5_INCLUDE_DIR})
    add_definitions(-DWITH_HDF5)
endif()



#-------------------------------------------------------------------------------------------------------------------
# find boost
#-------------------------------------------------------------------------------------------------------------------
find_package(Boost  REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
MESSAGE( STATUS ${Boost_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIR})
MESSAGE( STATUS ${Boost_INCLUDE_DIR})


if(WITHIN_TRAVIS)
    add_definitions(-DWITHIN_TRAVIS)
endif()


#-------------------------------------------------------------------------------------------------------------------
# find LP SOLVERS
#-------------------------------------------------------------------------------------------------------------------

SET(NIFTY_LP_SOLVER_LIBS "")
#-------------------------------------------------------------------------------------------------------------------
# find cplex
#-------------------------------------------------------------------------------------------------------------------
if(WITH_CPLEX)
    find_package(CPLEX REQUIRED)
    include_directories(${CPLEX_INCLUDE_DIRS})
    add_definitions(-DWITH_CPLEX)
    SET(NIFTY_LP_SOLVER_LIBS "${NIFTY_LP_SOLVER_LIBS};${CPLEX_LIBRARIES}")
endif()

#-------------------------------------------------------------------------------------------------------------------
# find gurobi
#-------------------------------------------------------------------------------------------------------------------
if(WITH_GUROBI)
    find_package(GUROBI REQUIRED)
    include_directories(${GUROBI_INCLUDE_DIR})
    add_definitions(-DWITH_GUROBI)
    SET(NIFTY_LP_SOLVER_LIBS "${NIFTY_LP_SOLVER_LIBS};${GUROBI_LIBRARIES}")
endif()

#-------------------------------------------------------------------------------------------------------------------
# find glpk
#-------------------------------------------------------------------------------------------------------------------
if(WITH_GLPK)
    find_package(GLPK REQUIRED)
    include_directories(${GLPK_INCLUDE_DIR})
    add_definitions(-DWITH_GLPK)
    SET(NIFTY_LP_SOLVER_LIBS "${NIFTY_LP_SOLVER_LIBS};${GLPK_LIBRARY}")
endif()






#-------------------------------------------------------------------------------------------------------------------
# find fastfilters
#-------------------------------------------------------------------------------------------------------------------
if(WITH_FASTFILTERS)
    find_package(FASTFILTERS REQUIRED)
    include_directories(${FASTFILTERS_INCLUDE_DIR})
    add_definitions(-DWITH_FASTFILTERS)
endif()


#-------------------------------------------------------------------------------------------------------------------
# additional dependencies for LP_MP
#-------------------------------------------------------------------------------------------------------------------
if(WITH_LP_MP)


    find_package(HDF5 REQUIRED)
    include_directories(${HDF5_INCLUDE_DIR})
    add_definitions(-DWITH_HDF5)

    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/LP_MP/include")
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/LP_MP/lib")
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/externals/LP_MP/lib")

    add_definitions(-DWITH_LP_MP)
    add_definitions(-DWITH_SAT)

    if(OPENMP_FOUND)
        add_definitions(-DLP_MP_PARALLEL)
    endif(OPENMP_FOUND)

    # set additional flags for the release build
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        # adding only -ffast-math will result in infinity and nan not being checked (but e.g. graph matching and discrete tomography have infinite costs)
        #add_definitions(-ffast-math -fno-finite-math-only)
        add_definitions()
        SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
        SET(CMAKE_EXE_LINKER_FLAGS_RELEASE  "${CMAKE_EXE_LINKER_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS_RELEASE}" )
    endif()

    # include LP_MP dependencies
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/LP_MP/external/meta/include")
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/LP_MP/external/Catch/include")
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/LP_MP/external/cpp_sort/include")
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/LP_MP/external/opengm/include")
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/LP_MP/external/PEGTL")
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/LP_MP/external/cereal/include")
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/LP_MP/external/tclap/include")
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/LP_MP/external/lingeling-ala")

    ## TODO this seems to be a pretty old version, would the newer version help?!
    ## FIXME we actually don't need to link this for building the LP_MP multicut
    #include(ExternalProject)
    #ExternalProject_ADD(
    #   lingeling
    #   #SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/lingeling-ala"
    #   URL "http://fmv.jku.at/lingeling/lingeling-bal-2293bef-151109.tar.gz"
    #   BUILD_IN_SOURCE 1
    #   #BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/lingeling-ala"
    #   CONFIGURE_COMMAND "./configure.sh"
    #   BUILD_COMMAND "make"
    #   INSTALL_COMMAND ""
    #   )
    #include_directories("${CMAKE_CURRENT_BINARY_DIR}/lingeling-prefix/src/lingeling")
    ##link_directories("${CMAKE_CURRENT_BINARY_DIR}/lingeling-prefix/src/lingeling")

endif()


if(WITH_QPBO)
    add_definitions(-DWITH_QPBO)
endif(WITH_QPBO)


#-------------------------------------------------------------------------------------------------------------------
# enable_testing
#-------------------------------------------------------------------------------------------------------------------
enable_testing()

#-------------------------------------------------------------------------------------------------------------------
# global headers
#-------------------------------------------------------------------------------------------------------------------
file(GLOB_RECURSE headers include/*.hxx)
file(GLOB_RECURSE headers ${CMAKE_INSTALL_PREFIX}/include/*.hxx)
include_directories(include)


#-------------------------------------------------------------------------------------------------------------------
# src folder
#-------------------------------------------------------------------------------------------------------------------
add_subdirectory(src)


#-------------------------------------------------------------------------------------------------------------------
# docsrc folder
#-------------------------------------------------------------------------------------------------------------------
add_subdirectory(docsrc)


















#-------------------------------------------------------------------------------------------------------------------
#  install headers
#-------------------------------------------------------------------------------------------------------------------
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/nifty"
  DESTINATION include
  FILES_MATCHING
  PATTERN "*.hxx"
  PATTERN "*.hpp"
  PATTERN "*.h"
)
