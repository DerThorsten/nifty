



# Create the binding library
pybind11_add_module(_graph 
    graph.cxx
    undirected_list_graph.cxx
    edge_contraction_graph_undirected_graph.cxx
    # gala
    gala/gala.cxx
    gala/gala_main_class.cxx
    gala/gala_feature_base.cxx
    #############################
    # multicut
    #############################
    multicut/multicut.cxx
    multicut/multicut_objective.cxx
    multicut/multicut_visitor_base.cxx
    multicut/multicut_base.cxx
    multicut/multicut_factory.cxx
    multicut/multicut_ilp.cxx
    multicut/multicut_greedy_additive.cxx
    multicut/fusion_move_based.cxx
    multicut/perturb_and_map.cxx
    #############################
    # lifted -multicut
    #############################
    lifted_multicut/lifted_multicut.cxx
    lifted_multicut/lifted_multicut_objective.cxx
    lifted_multicut/lifted_multicut_visitor_base.cxx
    lifted_multicut/lifted_multicut_base.cxx
    lifted_multicut/lifted_multicut_factory.cxx
    lifted_multicut/lifted_multicut_greedy_additive.cxx
    lifted_multicut/lifted_multicut_kernighan_lin.cxx
    lifted_multicut/lifted_multicut_ilp.cxx
    lifted_multicut/lifted_multicut_andres_kernighan_lin.cxx
    lifted_multicut/lifted_multicut_andres_greedy_additive.cxx
    lifted_multicut/lifted_graph_features.cxx
)


if(WITH_HDF5)
    target_link_libraries(_graph ${HDF5_LIBRARIES})
endif()

if(WITH_GUROBI)
    target_link_libraries(_graph ${GUROBI_LIBRARIES})
endif()

if(WITH_CPLEX)
    target_link_libraries(_graph ${CPLEX_LIBRARIES})
endif()

if(WITH_GLPK)
    target_link_libraries(_graph ${GLPK_LIBRARY})
endif()


# Don't add a 'lib' prefix to the shared library
set_target_properties(_graph PROPERTIES PREFIX "")

# Link against the Python shared library
target_link_libraries(_graph ${PYTHON_LIBRARIES})




file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/python/nifty/graph)

ADD_CUSTOM_COMMAND(
        TARGET _graph
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different $<TARGET_FILE:_graph> ${CMAKE_BINARY_DIR}/python/nifty/graph
        COMMENT "Copying target _graph to temporary module directory")




add_custom_target(copy_python_graph__init__ ALL#POST_BUILD
    COMMAND cmake -E copy_if_different 
    ${CMAKE_SOURCE_DIR}/src/python/module/graph/__init__.py 
    ${CMAKE_BINARY_DIR}/python/nifty/graph/__init__.py)



# submodules
add_subdirectory(agglo)
add_subdirectory(rag)