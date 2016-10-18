



# Create the binding library
pybind11_add_module(_structured_learning 
    structured_learning.cxx
    struct_max_margin.cxx
    struct_max_margin_oracle_lmc.cxx
)


if(WITH_CPLEX)
    target_link_libraries(_structured_learning ${CPLEX_LIBRARIES})
endif()

if(WITH_GLPK)
    target_link_libraries(_structured_learning ${GLPK_LIBRARY})
endif()




# Don't add a 'lib' prefix to the shared library
set_target_properties(_structured_learning PROPERTIES PREFIX "")

# Link against the Python shared library
target_link_libraries(_structured_learning ${PYTHON_LIBRARIES})




file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/python/nifty/structured_learning)

ADD_CUSTOM_COMMAND(
        TARGET _structured_learning
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different $<TARGET_FILE:_structured_learning> ${CMAKE_BINARY_DIR}/python/nifty/structured_learning
        COMMENT "Copying target _structured_learning to temporary module directory")




add_custom_target(copy_python_structured_learning__init__ ALL#POST_BUILD
    COMMAND cmake -E copy_if_different 
    ${CMAKE_SOURCE_DIR}/src/python/module/structured_learning/__init__.py 
    ${CMAKE_BINARY_DIR}/python/nifty/structured_learning/__init__.py)



