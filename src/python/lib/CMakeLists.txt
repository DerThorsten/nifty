

###########################################################################
# create a custom target which bundles all python building and copying
###########################################################################
add_custom_target(python-module)


###########################################################################
# copy all python files from:
#    src/python/module 
# recursively into 
#   /blddir/python/
###########################################################################
file(GLOB_RECURSE  ${MODULE_NAME}_PYTHON_FILES  
    "${CMAKE_SOURCE_DIR}/src/python/module/nifty/*.py"
)
FOREACH(PYTHON_FILENAME   ${${MODULE_NAME}_PYTHON_FILES})
    string(REPLACE "${CMAKE_SOURCE_DIR}/src/python/module/" "" 
        RELATIVE_FILENAME
        ${PYTHON_FILENAME})
    string(REPLACE "/" "_" 
        RELATIVE_FILENAME_TARGET_NAME
        ${RELATIVE_FILENAME})
    get_filename_component(FILE_DIR  ${CMAKE_BINARY_DIR}/${RELATIVE_FILENAME} DIRECTORY)
    file(MAKE_DIRECTORY ${FILE_DIR})

    add_custom_target(copy_${RELATIVE_FILENAME_TARGET_NAME} ALL
        COMMAND cmake -E copy_if_different 
        ${PYTHON_FILENAME}
        ${CMAKE_BINARY_DIR}/python/${RELATIVE_FILENAME})
    add_dependencies(python-module copy_${RELATIVE_FILENAME_TARGET_NAME})
ENDFOREACH()


###########################################################################
# macro to simplyfy the adding of a submodule
###########################################################################
macro(addPythonModule)

    set(options "")
    set(oneValueArgs NESTED_NAME)
    set(multiValueArgs SOURCES LIBRRARIES)
    cmake_parse_arguments(ADD_PY_MOD "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    message(STATUS ${LIBRRARIES})
    #message(STATUS "BRAAA"  ${ADD_PY_MOD_NESTED_NAME})

    # get name of the module
    string(REPLACE "/" ";" MOD_NESTING_LIST "${ADD_PY_MOD_NESTED_NAME}")
    list(LENGTH MOD_NESTING_LIST nesting_len)
    LIST(GET MOD_NESTING_LIST ${LENGTH}-1 MODULE_NAME)


    #Create the binding library
    pybind11_add_module(_${MODULE_NAME} 
        ${ADD_PY_MOD_SOURCES}
    )

    set_target_properties(_${MODULE_NAME}  PROPERTIES PREFIX "")

    message(STATUS ********Module: ${MODULE_NAME})
    message(STATUS ********Libs: ${ADD_PY_MOD_LIBRRARIES})

    # link additional libraries
    target_link_libraries(_${MODULE_NAME} 
        ${PYTHON_LIBRARIES} 
        ${ADD_PY_MOD_LIBRRARIES}
    )

    # create module dir
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/python/${ADD_PY_MOD_NESTED_NAME})

    # copy the module file
    ADD_CUSTOM_COMMAND(
        TARGET _${MODULE_NAME}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different $<TARGET_FILE:_${MODULE_NAME}> 
        ${CMAKE_BINARY_DIR}/python/${ADD_PY_MOD_NESTED_NAME}/
        COMMENT "Copying target ${MODULE_NAME} to temporary module directory")

    add_dependencies(python-module _${MODULE_NAME})
endmacro()




addPythonModule( NESTED_NAME nifty  SOURCES nifty.cxx )


# submodules
add_subdirectory(graph)
add_subdirectory(hdf5)
add_subdirectory(ground_truth)
add_subdirectory(pipelines)
add_subdirectory(tools)
add_subdirectory(ufd)
add_subdirectory(cgp)
add_subdirectory(malis)

