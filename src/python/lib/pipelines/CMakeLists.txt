



# Create the binding library
pybind11_add_module(_pipelines 
    pipelines.cxx
)


if(WITH_HDF5)
    target_link_libraries(_pipelines ${HDF5_LIBRARIES})
endif()


# Don't add a 'lib' prefix to the shared library
set_target_properties(_pipelines PROPERTIES PREFIX "")

# Link against the Python shared library
target_link_libraries(_pipelines ${PYTHON_LIBRARIES})




file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/python/nifty/pipelines)

ADD_CUSTOM_COMMAND(
        TARGET _pipelines
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different $<TARGET_FILE:_pipelines> ${CMAKE_BINARY_DIR}/python/nifty/pipelines
        COMMENT "Copying target _pipelines to temporary module directory")




add_custom_target(copy_python_pipelines__init__ ALL#POST_BUILD
    COMMAND cmake -E copy_if_different 
    ${CMAKE_SOURCE_DIR}/src/python/module/pipelines/__init__.py 
    ${CMAKE_BINARY_DIR}/python/nifty/pipelines/__init__.py)


