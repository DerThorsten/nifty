



# Create the binding library
pybind11_add_module(_lifted_multicut 
    # rag
    lifted_multicut.cxx
    lifted_multicut_objective.cxx
    weighted_lifted_multicut_objective.cxx
    loss_augmented_view_lifted_multicut_objective.cxx
    lifted_multicut_visitor_base.cxx
    lifted_multicut_base.cxx
    lifted_multicut_factory.cxx
    lifted_multicut_greedy_additive.cxx
    lifted_multicut_kernighan_lin.cxx
    lifted_multicut_ilp.cxx
    lifted_multicut_andres_kernighan_lin.cxx
    lifted_multicut_andres_greedy_additive.cxx
    fusion_move_based.cxx
    chained_solver.cxx
    lifted_graph_features.cxx
)



if(WITH_GUROBI)
    target_link_libraries(_lifted_multicut ${GUROBI_LIBRARIES})
endif()

if(WITH_CPLEX)
    target_link_libraries(_lifted_multicut ${CPLEX_LIBRARIES})
endif()

if(WITH_GLPK)
    target_link_libraries(_lifted_multicut ${GLPK_LIBRARY})
endif()


# Don't add a 'lib' prefix to the shared library
set_target_properties(_lifted_multicut PROPERTIES PREFIX "")

# Link against the Python shared library
target_link_libraries(_lifted_multicut ${PYTHON_LIBRARIES})




file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/python/nifty/graph/lifted_multicut)

ADD_CUSTOM_COMMAND(
        TARGET _lifted_multicut
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different $<TARGET_FILE:_lifted_multicut> ${CMAKE_BINARY_DIR}/python/nifty/graph/lifted_multicut
        COMMENT "Copying target lifted_multicut to temporary module directory")




add_custom_target(copy_pythonlifted_multicut__init__ ALL#POST_BUILD
    COMMAND cmake -E copy_if_different 
    ${CMAKE_SOURCE_DIR}/src/python/module/graph/lifted_multicut/__init__.py 
    ${CMAKE_BINARY_DIR}/python/nifty/graph/lifted_multicut/__init__.py)



