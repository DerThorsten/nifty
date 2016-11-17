

# create a custom target which bundles all python building and copying
add_custom_target(python-module)



# Create the binding library
pybind11_add_module(_nifty 
    nifty.cxx
    #############################
    # ufd
    #############################
    ufd/ufd.cxx
)





file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/python/nifty)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/python/nifty/viewer)
 

ADD_CUSTOM_COMMAND(
        TARGET _nifty
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different $<TARGET_FILE:_nifty> ${CMAKE_BINARY_DIR}/python/nifty/
        COMMENT "Copying target _nifty to temporary module directory")

add_custom_target(copy_python__init__ ALL#POST_BUILD
    COMMAND cmake -E copy_if_different 
    ${CMAKE_SOURCE_DIR}/src/python/module/__init__.py 
    ${CMAKE_BINARY_DIR}/python/nifty/__init__.py)


add_custom_target(copy_python_viewer__init__ ALL#POST_BUILD
    COMMAND cmake -E copy_if_different 
    ${CMAKE_SOURCE_DIR}/src/python/module/viewer/__init__.py 
    ${CMAKE_BINARY_DIR}/python/nifty/viewer/__init__.py)






add_dependencies(python-module _nifty)
add_dependencies(python-module copy_python__init__)
add_dependencies(python-module copy_python_viewer__init__)


# submodules
add_subdirectory(graph)
add_subdirectory(hdf5)
add_subdirectory(ground_truth)
add_subdirectory(pipelines)
add_subdirectory(tools)
