



# Create the binding library
pybind11_add_module(_nifty 
    nifty.cxx
    #############################
    # graph module
    #############################
    # ... extra files go here ...
    graph/graph.cxx
    graph/undirected_list_graph.cxx
    graph/edge_contraction_graph_undirected_graph.cxx
    # rag
    graph/rag/rag.cxx
    graph/rag/grid_rag.cxx
    graph/rag/graph_accumulator.cxx
    graph/rag/project_to_pixels.cxx
    graph/rag/accumulate.cxx
    # agglo
    graph/agglo/agglo.cxx
    graph/agglo/agglomerative_clustering.cxx
    # gala
    graph/gala/gala.cxx
    graph/gala/gala_main_class.cxx
    graph/gala/gala_feature_base.cxx
    #############################
    # multicut
    #############################
    graph/multicut/multicut.cxx
    graph/multicut/multicut_objective.cxx
    graph/multicut/multicut_visitor_base.cxx
    graph/multicut/multicut_base.cxx
    graph/multicut/multicut_factory.cxx
    graph/multicut/multicut_ilp.cxx
    graph/multicut/multicut_greedy_additive.cxx
    graph/multicut/fusion_move_based.cxx
    graph/multicut/perturb_and_map.cxx
    #############################
    # lifted -multicut
    #############################
    graph/lifted_multicut/lifted_multicut.cxx
    graph/lifted_multicut/lifted_multicut_objective.cxx
    graph/lifted_multicut/lifted_multicut_visitor_base.cxx
    graph/lifted_multicut/lifted_multicut_base.cxx
    graph/lifted_multicut/lifted_multicut_factory.cxx
    graph/lifted_multicut/lifted_multicut_greedy_additive.cxx
    graph/lifted_multicut/lifted_multicut_kernighan_lin.cxx
    graph/lifted_multicut/lifted_multicut_ilp.cxx
    #############################
    # hdf5
    #############################
    tools/tools.cxx
    tools/make_dense.cxx
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
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/test_python/)

ADD_CUSTOM_COMMAND(
        TARGET _nifty
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different $<TARGET_FILE:_nifty> ${CMAKE_BINARY_DIR}/python/nifty/
        COMMENT "Copying target _graph to temporary module directory")

add_custom_target(copy_python__init__ ALL#POST_BUILD
    COMMAND cmake -E copy_if_different 
    ${CMAKE_SOURCE_DIR}/src/python/module/__init__.py 
    ${CMAKE_BINARY_DIR}/python/nifty/__init__.py)



add_custom_target(copy_python_test ALL#POST_BUILD
    COMMAND cmake -E copy_if_different 
    ${CMAKE_SOURCE_DIR}/src/python/test/test_python.py 
    ${CMAKE_BINARY_DIR}/test_python/test_python.py)

