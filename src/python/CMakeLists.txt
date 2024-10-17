find_package(xtensor-python REQUIRED)
include_directories(${xtensor-python_INCLUDE_DIRS})

find_package(pybind11 REQUIRED)
include_directories(${pybind11_INCLUDE_DIRS})

if(MSVC)
    set(PYBIND11_CPP_STANDARD /std:c++17)
else()
    set(PYBIND11_CPP_STANDARD -std=c++17)
endif()
message(STATUS "Using c++ 17 for pybind")

find_package(Python 3 REQUIRED COMPONENTS NumPy)
include_directories(${Python_NumPy_INCLUDE_DIRS})

add_subdirectory(test)
add_subdirectory(lib)

install(DIRECTORY ${CMAKE_BINARY_DIR}/python/nifty
    DESTINATION ${Python_SITELIB})
