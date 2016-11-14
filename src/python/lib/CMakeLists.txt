

# create a custom target which bundles all python building and copying
add_custom_target(python-module)



# Create the binding library
pybind11_add_module(_nifty 
    nifty.cxx

    #############################
    # hdf5
    #############################
    hdf5/hdf5.cxx
    hdf5/hdf5_common.cxx
    hdf5/hdf5_array.cxx
)




if(WITH_HDF5)
    target_link_libraries(_nifty ${HDF5_LIBRARIES})
endif()

if(WITH_GUROBI)
    target_link_libraries(_nifty ${GUROBI_LIBRARIES})
endif()

if(WITH_CPLEX)
    target_link_libraries(_nifty ${CPLEX_LIBRARIES})
endif()

if(WITH_GLPK)
    target_link_libraries(_nifty ${GLPK_LIBRARY})
endif()


# Don't add a 'lib' prefix to the shared library
set_target_properties(_nifty PROPERTIES PREFIX "")

# Link against the Python shared library
target_link_libraries(_nifty ${PYTHON_LIBRARIES})




file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/python/nifty)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/python/nifty/viewer)
 

ADD_CUSTOM_COMMAND(
        TARGET _nifty
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different $<TARGET_FILE:_nifty> ${CMAKE_BINARY_DIR}/python/nifty/
        COMMENT "Copying target _nifty to temporary module directory")

add_custom_target(copy_python__init__ ALL#POST_BUILD
    COMMAND cmake -E copy_if_different 
    ${CMAKE_SOURCE_DIR}/src/python/module/__init__.py 
    ${CMAKE_BINARY_DIR}/python/nifty/__init__.py)


add_custom_target(copy_python_viewer__init__ ALL#POST_BUILD
    COMMAND cmake -E copy_if_different 
    ${CMAKE_SOURCE_DIR}/src/python/module/viewer/__init__.py 
    ${CMAKE_BINARY_DIR}/python/nifty/viewer/__init__.py)






add_dependencies(python-module _nifty)
add_dependencies(python-module copy_python__init__)
add_dependencies(python-module copy_python_viewer__init__)


# submodules
add_subdirectory(graph)
add_subdirectory(ground_truth)
add_subdirectory(pipelines)
add_subdirectory(tools)
