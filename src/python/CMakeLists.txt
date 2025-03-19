find_package(Python REQUIRED COMPONENTS NumPy Interpreter Development)
include_directories(${Python_NumPy_INCLUDE_DIRS})

find_package(xtensor-python REQUIRED)
include_directories(${xtensor-python_INCLUDE_DIRS})

find_package(pybind11 CONFIG REQUIRED)
include_directories(${pybind11_INCLUDE_DIRS})

add_subdirectory(test)
add_subdirectory(lib)

install(DIRECTORY ${CMAKE_BINARY_DIR}/python/nifty
    DESTINATION ${Python_SITELIB})
