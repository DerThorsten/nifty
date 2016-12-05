



# Create the binding library
pybind11_add_module(_agglo 
    # agglo
    agglo.cxx
    agglomerative_clustering.cxx
)


if(WITH_HDF5)
    target_link_libraries(_agglo ${HDF5_LIBRARIES})
endif()

if(WITH_GUROBI)
    target_link_libraries(_agglo ${GUROBI_LIBRARIES})
endif()

if(WITH_CPLEX)
    target_link_libraries(_agglo ${CPLEX_LIBRARIES})
endif()

if(WITH_GLPK)
    target_link_libraries(_agglo ${GLPK_LIBRARY})
endif()


# Don't add a 'lib' prefix to the shared library
set_target_properties(_agglo PROPERTIES PREFIX "")

# Link against the Python shared library
target_link_libraries(_agglo ${PYTHON_LIBRARIES})




file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/python/nifty/graph/agglo)

ADD_CUSTOM_COMMAND(
        TARGET _agglo
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different $<TARGET_FILE:_agglo> ${CMAKE_BINARY_DIR}/python/nifty/graph/agglo
        COMMENT "Copying target agglo to temporary module directory")




add_custom_target(copy_pythonagglo__init__ ALL#POST_BUILD
    COMMAND cmake -E copy_if_different 
    ${CMAKE_SOURCE_DIR}/src/python/module/graph/agglo/__init__.py 
    ${CMAKE_BINARY_DIR}/python/nifty/graph/agglo/__init__.py)


add_dependencies(python-module _agglo)
add_dependencies(python-module copy_pythonagglo__init__)



