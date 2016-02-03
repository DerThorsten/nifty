cmake_minimum_required(VERSION 2.8)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

project(nifty)


include(CheckCXXCompilerFlag)


option(WITH_BOOST_COMPUTE OFF)
option(BUILD_NIFTY_PYTHON OFF)


#-------------------------------------------------------------------------------------------------------------------
# check for c++ 11 support
#-------------------------------------------------------------------------------------------------------------------
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


#-------------------------------------------------------------------------------------------------------------------
# Set a default build configuration if none is specified. 'MinSizeRel' produces the smallest binaries
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

if(WITH_BOOST_COMPUTE)
    #----------------------------------------------------------------------------------------------------------------
    # find OpenGL
    #----------------------------------------------------------------------------------------------------------------
    find_package(OpenCL REQUIRED)
    include_directories(${OpenCL_INCLUDE_DIRS})

    #----------------------------------------------------------------------------------------------------------------
    # find BOOST_COMPUTE
    #----------------------------------------------------------------------------------------------------------------
    find_package(BOOST_COMPUTE REQUIRED)
    include_directories(${BOOST_COMPUTE_INCLUDE_DIR})
endif()
#-------------------------------------------------------------------------------------------------------------------
# find boost
#-------------------------------------------------------------------------------------------------------------------
find_package(Boost  REQUIRED)


# #-------------------------------------------------------------------------------------------------------------------
# # find vigra
# #-------------------------------------------------------------------------------------------------------------------
# find_package(VIGRA)



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
