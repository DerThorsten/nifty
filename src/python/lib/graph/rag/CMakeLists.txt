



# Create the binding library
pybind11_add_module(_rag 
    # rag
    rag.cxx
    grid_rag.cxx
    graph_accumulator.cxx
    project_to_pixels.cxx
    accumulate.cxx
    accumulate_stacked.cxx
    accumulate_filters_stacked.cxx
)


if(WITH_HDF5)
    target_link_libraries(_rag ${HDF5_LIBRARIES})
endif()

if(WITH_GUROBI)
    target_link_libraries(_rag ${GUROBI_LIBRARIES})
endif()

if(WITH_CPLEX)
    target_link_libraries(_rag ${CPLEX_LIBRARIES})
endif()

if(WITH_GLPK)
    target_link_libraries(_rag ${GLPK_LIBRARY})
endif()

if(WITH_FASTFILTERS)
    target_link_libraries(_rag ${FASTFILTERS_LIBRARY})
endif()

# Don't add a 'lib' prefix to the shared library
set_target_properties(_rag PROPERTIES PREFIX "")

# Link against the Python shared library
target_link_libraries(_rag ${PYTHON_LIBRARIES})




file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/python/nifty/graph/rag)

ADD_CUSTOM_COMMAND(
        TARGET _rag
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different $<TARGET_FILE:_rag> ${CMAKE_BINARY_DIR}/python/nifty/graph/rag
        COMMENT "Copying target rag to temporary module directory")




add_custom_target(copy_pythonrag__init__ ALL#POST_BUILD
    COMMAND cmake -E copy_if_different 
    ${CMAKE_SOURCE_DIR}/src/python/module/graph/rag/__init__.py 
    ${CMAKE_BINARY_DIR}/python/nifty/graph/rag/__init__.py)



add_dependencies(python-module _rag)
add_dependencies(python-module copy_pythonrag__init__)
