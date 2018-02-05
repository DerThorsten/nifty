
#message(PYTHON_VERSION_MAJOR ${PYTHON_VERSION_MAJOR})
if(PYTHON_VERSION_MAJOR STREQUAL "2")
    #message(STATUS "222222")
    find_program(NOSETESTS_PATH NAMES
            #"nosetests${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}"
            #"nosetests-${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}"
            #"nosetests${PYTHON_VERSION_MAJOR}"
            #"nosetests-${PYTHON_VERSION_MAJOR}"
            "nosetests")
else()
    find_program(NOSETESTS_PATH NAMES
        #"nosetests${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}"
        #"nosetests-${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}"
        #"nosetests${PYTHON_VERSION_MAJOR}"
        #"nosetests-${PYTHON_VERSION_MAJOR}"
        "nosetests3")
endif()

if(NOT NOSETESTS_PATH)
    message(WARNING 
        "nosetests not found! Python library tests will not be available.")
endif()


function(add_python_test_target TARGET_NAME)

    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/python_test)

    # Try again to find nosetests here. We may have switched virtualenvs or
    # something since first running cmake.
    find_program(NOSETESTS_PATH nosetests)
    if(NOT NOSETESTS_PATH)
        message(FATAL_ERROR "nosetests not found! Aborting...")
    endif()

    set(COPY_DIR ${CMAKE_BINARY_DIR}/python_test/${TARGET_NAME})


    set(COPY_MOD_TARGET CopyModuleDir${TARGET_NAME})
    add_custom_target(${COPY_MOD_TARGET} ALL)


    add_custom_command(TARGET ${COPY_MOD_TARGET} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_BINARY_DIR}/python/
        ${COPY_DIR}
    )


    add_dependencies(${COPY_MOD_TARGET} python-module)












    add_custom_target(${TARGET_NAME}
        COMMAND ${NOSETESTS_PATH}  --nologcapture
        WORKING_DIRECTORY ${COPY_DIR}
        COMMENT "Running Python tests.")




    add_dependencies(${TARGET_NAME} ${COPY_MOD_TARGET})


    # Copy Python files to the local binary directory so they can find the
    # dynamic library.
    set(COPY_TARGET ${TARGET_NAME}_copy)

    # We add a separate target for copying Python files. This way we can set
    # it as a dependency for the nosetests target.
    add_custom_target(${COPY_TARGET})

    # Make sure the directory exists before we copy the files.
    add_custom_command(TARGET ${COPY_TARGET}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${COPY_DIR})

    # Add commands to copy each file in the list of sources.
    foreach(pysource ${ARGN})
        add_custom_command(TARGET ${COPY_TARGET}
            COMMAND ${CMAKE_COMMAND} -E copy 
            ${CMAKE_CURRENT_SOURCE_DIR}/${pysource} ${COPY_DIR})
    endforeach()

    #get_target_property(TARGET_LIB_NAME ${TARGET_LIB} LOCATION)
    #$<TARGET_FILE:${TARGET_LIB}>

   

    # Make the copy target a dependency of the testing target to ensure it
    # gets done first.
    add_dependencies(${TARGET_NAME} ${COPY_TARGET})


endfunction()