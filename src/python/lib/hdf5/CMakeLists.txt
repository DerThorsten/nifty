



# Create the binding library
pybind11_add_module(_hdf5 
    hdf5.cxx
    hdf5_common.cxx
    hdf5_array.cxx
    hdf5_watershed.cxx
)

if(WITH_HDF5)
    target_link_libraries(_hdf5 ${HDF5_LIBRARIES})
endif()


# Don't add a 'lib' prefix to the shared library
set_target_properties(_hdf5 PROPERTIES PREFIX "")

# Link against the Python shared library
target_link_libraries(_hdf5 ${PYTHON_LIBRARIES})




file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/python/nifty/hdf5)

ADD_CUSTOM_COMMAND(
        TARGET _hdf5
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different $<TARGET_FILE:_hdf5> ${CMAKE_BINARY_DIR}/python/nifty/hdf5
        COMMENT "Copying target _hdf5 to temporary module directory")




add_custom_target(copy_python_hdf5__init__ ALL#POST_BUILD
    COMMAND cmake -E copy_if_different 
    ${CMAKE_SOURCE_DIR}/src/python/module/hdf5/__init__.py 
    ${CMAKE_BINARY_DIR}/python/nifty/hdf5/__init__.py)




add_dependencies(python-module _hdf5)
add_dependencies(python-module copy_python_hdf5__init__)