find_package(xtensor-python REQUIRED)
include_directories(${xtensor-python_INCLUDE_DIRS})

find_package(pybind11 REQUIRED)
include_directories(${pybind11_INCLUDE_DIRS})

if (CPP17)
    if(MSVC)
        set(PYBIND11_CPP_STANDARD /std:c++17)
    else()
        set(PYBIND11_CPP_STANDARD -std=c++17)
    endif()
    message(STATUS "Using c++ 17 for pybind")
else()
    if(MSVC)
        set(PYBIND11_CPP_STANDARD /std:c++14)
    else()
        set(PYBIND11_CPP_STANDARD -std=c++14)
    endif()
    message(STATUS "Using c++ 14 for pybind")
endif()

find_package(NUMPY REQUIRED)
include_directories(${NUMPY_INCLUDE_DIRS})


add_subdirectory(test)
add_subdirectory(lib)


#################################
# INSTALL THE PYTHON MODULE
#################################

IF(NOT DEFINED PYTHON_MODULE_INSTALL_DIR OR PYTHON_MODULE_INSTALL_DIR MATCHES "^$")


    execute_process(
      COMMAND "${PYTHON_EXECUTABLE}" -c 
       "from __future__ import print_function; from distutils import sysconfig as sc; print(sc.get_python_lib(prefix='', plat_specific=True))"
      OUTPUT_VARIABLE PYTHON_SITE
      OUTPUT_STRIP_TRAILING_WHITESPACE)

    execute_process(
      COMMAND "${PYTHON_EXECUTABLE}" -c "from __future__ import print_function; from distutils.sysconfig import get_python_lib; print(get_python_lib())"
      OUTPUT_VARIABLE PYTHON_SITE
      OUTPUT_STRIP_TRAILING_WHITESPACE)

    SET(PYTHON_MODULE_INSTALL_DIR ${PYTHON_SITE})

ENDIF()

SET(PYTHON_MODULE_INSTALL_DIR ${PYTHON_MODULE_INSTALL_DIR}
    CACHE PATH "where to install the Nifty Python package" FORCE)

# this is the install path relative to CMAKE_INSTALL_PREFIX,
# use this in INSTALL() commands to get packaging right
FILE(RELATIVE_PATH PYTHON_MODULE_INSTALL_DIR ${CMAKE_INSTALL_PREFIX} ${PYTHON_MODULE_INSTALL_DIR})



install(DIRECTORY ${CMAKE_BINARY_DIR}/python/nifty
    DESTINATION ${PYTHON_MODULE_INSTALL_DIR})
