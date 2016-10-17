



# Create the binding library
pybind11_add_module(_ground_truth 
    ground_truth.cxx
    overlap.cxx
    partition_comparison.cxx
    post_process_carving_ground_truth.cxx
)


# if(WITH_HDF5)
#     target_link_libraries(_ground_truth ${HDF5_LIBRARIES})
# endif()

# if(WITH_GUROBI)
#     target_link_libraries(_ground_truth ${GUROBI_LIBRARIES})
# endif()

# if(WITH_CPLEX)
#     target_link_libraries(_ground_truth ${CPLEX_LIBRARIES})
# endif()

# if(WITH_GLPK)
#     target_link_libraries(_ground_truth ${GLPK_LIBRARY})
# endif()


# Don't add a 'lib' prefix to the shared library
set_target_properties(_ground_truth PROPERTIES PREFIX "")

# Link against the Python shared library
target_link_libraries(_ground_truth ${PYTHON_LIBRARIES})




file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/python/nifty/ground_truth)

ADD_CUSTOM_COMMAND(
        TARGET _ground_truth
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different $<TARGET_FILE:_ground_truth> ${CMAKE_BINARY_DIR}/python/nifty/ground_truth
        COMMENT "Copying target _ground_truth to temporary module directory")




add_custom_target(copy_python_ground_truth__init__ ALL#POST_BUILD
    COMMAND cmake -E copy_if_different 
    ${CMAKE_SOURCE_DIR}/src/python/module/ground_truth/__init__.py 
    ${CMAKE_BINARY_DIR}/python/nifty/ground_truth/__init__.py)



