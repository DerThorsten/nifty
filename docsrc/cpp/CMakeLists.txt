#-------------------------------------------------------------------------------------------------------------------
# Doxygen Documentation
#-------------------------------------------------------------------------------------------------------------------

find_package(Doxygen)
if(BUILD_DOCS)
    if(DOXYGEN_FOUND)

        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in 
            ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY
        )

        # create output folder
        file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/docs/cpp)

        add_custom_target(cpp-doc
            ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            OUTPUT_DIRECTORY  ${CMAKE_BINARY_DIR}/docs/cpp
         COMMENT "Generating API documentation with Doxygen" VERBATIM
        )

        add_dependencies(doc cpp-doc)
    endif(DOXYGEN_FOUND)
endif(BUILD_DOCS)










