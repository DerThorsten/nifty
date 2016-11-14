



# Create the binding library
pybind11_add_module(_tools 
    tools.cxx
    make_dense.cxx
    blocking.cxx
)



# Don't add a 'lib' prefix to the shared library
set_target_properties(_tools PROPERTIES PREFIX "")

# Link against the Python shared library
target_link_libraries(_tools ${PYTHON_LIBRARIES})




file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/python/nifty/tools)

ADD_CUSTOM_COMMAND(
        TARGET _tools
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different $<TARGET_FILE:_tools> ${CMAKE_BINARY_DIR}/python/nifty/tools
        COMMENT "Copying target _tools to temporary module directory")




add_custom_target(copy_python_tools__init__ ALL#POST_BUILD
    COMMAND cmake -E copy_if_different 
    ${CMAKE_SOURCE_DIR}/src/python/module/tools/__init__.py 
    ${CMAKE_BINARY_DIR}/python/nifty/tools/__init__.py)





add_dependencies(python-module _tools)
add_dependencies(python-module copy_python_tools__init__)