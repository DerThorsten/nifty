



# Create the binding library
pybind11_add_module(_multicut 
    multicut.cxx
    multicut_objective.cxx
    multicut_visitor_base.cxx
    multicut_base.cxx
    multicut_factory.cxx
    multicut_ilp.cxx
    multicut_greedy_additive.cxx
    fusion_move_based.cxx
    perturb_and_map.cxx
)



if(WITH_GUROBI)
    target_link_libraries(_multicut ${GUROBI_LIBRARIES})
endif()

if(WITH_CPLEX)
    target_link_libraries(_multicut ${CPLEX_LIBRARIES})
endif()

if(WITH_GLPK)
    target_link_libraries(_multicut ${GLPK_LIBRARY})
endif()


# Don't add a 'lib' prefix to the shared library
set_target_properties(_multicut PROPERTIES PREFIX "")

# Link against the Python shared library
target_link_libraries(_multicut ${PYTHON_LIBRARIES})




file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/python/nifty/graph/multicut)

ADD_CUSTOM_COMMAND(
        TARGET _multicut
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different $<TARGET_FILE:_multicut> ${CMAKE_BINARY_DIR}/python/nifty/graph/multicut
        COMMENT "Copying target multicut to temporary module directory")




add_custom_target(copy_pythonmulticut__init__ ALL#POST_BUILD
    COMMAND cmake -E copy_if_different 
    ${CMAKE_SOURCE_DIR}/src/python/module/graph/multicut/__init__.py 
    ${CMAKE_BINARY_DIR}/python/nifty/graph/multicut/__init__.py)



