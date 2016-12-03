



# Create the binding library
pybind11_add_module(_ufd 
    ufd.cxx
    export_ufd.cxx
)

# Don't add a 'lib' prefix to the shared library
set_target_properties(_ufd PROPERTIES PREFIX "")

# Link against the Python shared library
target_link_libraries(_ufd ${PYTHON_LIBRARIES})




file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/python/nifty/ufd)

ADD_CUSTOM_COMMAND(
        TARGET _ufd
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different $<TARGET_FILE:_ufd> ${CMAKE_BINARY_DIR}/python/nifty/ufd
        COMMENT "Copying target _ufd to temporary module directory")




add_custom_target(copy_python_ufd__init__ ALL#POST_BUILD
    COMMAND cmake -E copy_if_different 
    ${CMAKE_SOURCE_DIR}/src/python/module/ufd/__init__.py 
    ${CMAKE_BINARY_DIR}/python/nifty/ufd/__init__.py)




add_dependencies(python-module _ufd)
add_dependencies(python-module copy_python_ufd__init__)
