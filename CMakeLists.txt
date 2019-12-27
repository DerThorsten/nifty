cmake_minimum_required(VERSION 3.1)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

project(NIFTY)
set(${PROJECT_NAME}_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)




#-------------------------------------------------------------------------------------------------------------------
# Version
#-------------------------------------------------------------------------------------------------------------------
# as seen in the github.com/QuantStack/xtensor
file(STRINGS "${${PROJECT_NAME}_INCLUDE_DIR}/nifty/nifty_config.hxx" ${PROJECT_NAME}_version_defines
     REGEX "#define ${PROJECT_NAME}_VERSION_(MAJOR|MINOR|PATCH)")
foreach(ver ${${PROJECT_NAME}_version_defines})
    if(ver MATCHES "#define ${PROJECT_NAME}_VERSION_(MAJOR|MINOR|PATCH) +([^ ]+)$")
        set(${PROJECT_NAME}_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}" CACHE INTERNAL "")
    endif()
endforeach()
set(${PROJECT_NAME}_VERSION 
    ${${PROJECT_NAME}_VERSION_MAJOR}.${${PROJECT_NAME}_VERSION_MINOR}.${${PROJECT_NAME}_VERSION_PATCH})
message(STATUS "${PROJECT_NAME} v${${PROJECT_NAME}_VERSION}")


include(CheckCXXCompilerFlag)

include_directories(${${PROJECT_NAME}_INCLUDE_DIR})

option(USE_OLD_ABI "Use old GLIBC ABI for backwards compatability" OFF)
# Old ABI to be compatible with boost
if(USE_OLD_ABI)
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
endif()

option(WITH_GUROBI "Build with gurobi" OFF)
option(WITH_CPLEX "Build with cplex" OFF)
option(WITH_GLPK "Build with glpk" OFF)
option(WITH_HDF5 "Build with hdf5 support" OFF)
option(WITH_FASTFILTERS "Build with fastfilters" OFF)
option(WITH_LP_MP "Build with lpmp" OFF)
option(WITH_QPBO "Build with qpbo" OFF)
option(WITH_Z5 "Build with z5" OFF)

option(BUILD_PYTHON_TEST "Build python tests" OFF)
option(BUILD_CPP_TEST "Build c++ tests" OFF)
option(BUILD_NIFTY_PYTHON "Build the python bindings" ON)
option(BUILD_DOCS "Build the docs" OFF)
option(BUILD_CPP_EXAMPLES "Build the c++ examples" OFF)
option(BUILD_PYTHON_DOCS "Build the Python documentation with Sphinx" OFF)

option(REMOVE_SOME_WARNINGS "Remove some annoying warnings" ON)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)


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


# boost geometry needs this 
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_definitions(-fext-numeric-literals)
endif()

#-------------------------------------------------------------------------------------------------------------------
# check for c++ 14 / 17 support
#-------------------------------------------------------------------------------------------------------------------

message(STATUS "CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
if (NOT "${CMAKE_CXX_FLAGS}" MATCHES "-std=c\\+\\+17")
    message(STATUS "Using c++ 14")
    set(CMAKE_CXX_STANDARD 14)
    set(CPP17 FALSE)
else()
    message(STATUS "Using c++ 17")
    set(CMAKE_CXX_STANDARD 17)
    set(CPP17 TRUE)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# WITH_LP_MP needs special treatment since
# it requires C++ 14 and uses OpenMP
if(WITH_LP_MP)
    find_package(OpenMP)
    if (OPENMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endif()


#-------------------------------------------------------------------------------------------------------------------
# Set a default build configuraton if none is specified. 'MinSizeRel' produces the smallest binaries
#-------------------------------------------------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
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


find_package(xtensor REQUIRED)
include_directories(${xtensor_INCLUDE_DIRS})


#-------------------------------------------------------------------------------------------------------------------
# externals
#-------------------------------------------------------------------------------------------------------------------
# FIXME external maxflow project doews not exist anymore
# was this removed accidentally?
# include_directories( "${CMAKE_CURRENT_SOURCE_DIR}/externals/maxflow")




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

# If WITH_BOOST_FS is set, we always use the boost filesystem library
# otherwise, we check if we have c++17 support. If so, we use std::filesystem,
# otherwise we fall back to boost filesystem.
if(CPP17)
    SET(WITH_BOOST_FS FALSE CACHE BOOL "")
else()
    SET(WITH_BOOST_FS TRUE CACHE BOOL "")
endif()

if(WITH_BOOST_FS)
    message(STATUS "With boost filesystem")
    find_package(Boost 1.63.0 COMPONENTS system filesystem REQUIRED)
    add_definitions(-DWITH_BOOST_FS)
    SET(FILESYSTEM_LIBRARIES "${Boost_FILESYSTEM_LIBRARY};${Boost_SYSTEM_LIBRARY}")
else()
    message(STATUS "With std filesystem")
    find_package(Boost 1.63.0 REQUIRED)

    # see this issue for discussions about the filesystem lib in CMake
    # https://gitlab.kitware.com/cmake/cmake/issues/17834
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        SET(FILESYSTEM_LIBRARIES "stdc++fs")
    endif()

    # on clang, we need to enable libc++experimental, see
    # https://stackoverflow.com/a/45332844
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        SET(FILESYSTEM_LIBRARIES "c++experimental")
    endif()
endif()
include_directories(${Boost_INCLUDE_DIR})


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
# find z5
#-------------------------------------------------------------------------------------------------------------------
if(WITH_Z5)
    add_definitions(-DWITH_Z5)
    option(WITH_BLOSC "Build z5 with blosc compression" ON)
    option(WITH_BZIP2 "Build z5 with bzip2 compression" ON)
    option(WITH_ZLIB "Build z5 with zlib compression" ON)

    # z5 compression libraries
    SET(Z5_COMPRESSION_LIBRARIES "")
    if(WITH_BLOSC)
        find_package(BLOSC REQUIRED)
        include_directories(${BLOSC_INCLUDE_DIR})
        add_definitions(-DWITH_BLOSC)
        SET(Z5_COMPRESSION_LIBRARIES "${Z5_COMPRESSION_LIBRARIES};${BLOSC_LIBRARIES}")
    endif()

    if(WITH_ZLIB)
        find_package(ZLIB REQUIRED)
        include_directories(ZLIB_INCLUDE_DIRS)
        add_definitions(-DWITH_ZLIB)
        SET(Z5_COMPRESSION_LIBRARIES "${Z5_COMPRESSION_LIBRARIES};${ZLIB_LIBRARIES}")
    endif()

    if(WITH_BZIP2)
        find_package(BZip2 REQUIRED)
        include_directories(BZIP2_INCLUDE_DIRS)
        add_definitions(-DWITH_BZIP2)
        SET(Z5_COMPRESSION_LIBRARIES "${Z5_COMPRESSION_LIBRARIES};${BZIP2_LIBRARIES}")
    endif()
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
    include_directories( "${CMAKE_CURRENT_SOURCE_DIR}/externals/qpbo")
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


#------------------------------------------------------------------------------------------------------------------
# install
#-------------------------------------------------------------------------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)


install(DIRECTORY  ${${PROJECT_NAME}_INCLUDE_DIR}/nifty
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/nifty)


set(${PROJECT_NAME}_CMAKECONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}" CACHE
    STRING "install path for ${PROJECT_NAME}Config.cmake")

configure_package_config_file(${PROJECT_NAME}Config.cmake.in
                              "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
                              INSTALL_DESTINATION ${${PROJECT_NAME}_CMAKECONFIG_INSTALL_DIR})


write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
                                 VERSION ${${PROJECT_NAME}_VERSION}
                                 COMPATIBILITY AnyNewerVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
        DESTINATION ${${PROJECT_NAME}_CMAKECONFIG_INSTALL_DIR})
