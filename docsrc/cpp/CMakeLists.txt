#-------------------------------------------------------------------------------------------------------------------
# Doxygen Documentation
#-------------------------------------------------------------------------------------------------------------------

find_package(Doxygen)
if(BUILD_DOCS)
    if(DOXYGEN_FOUND)

        file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/docs/cpp)

        # ordinary build
        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in 
            ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY
        )
        # create output folder
        
        add_custom_target(cpp-doc
            ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            OUTPUT_DIRECTORY  ${CMAKE_BINARY_DIR}/docs/cpp
         COMMENT "Generating API documentation with Doxygen" VERBATIM
        )
        add_dependencies(doc cpp-doc)



        if(BUILD_NIFTY_PYTHON AND BUILD_PYTHON_DOCS)
            # xml build for breathe
            configure_file(
                ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile_xml.in 
                ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile_xml @ONLY
            )
            add_custom_target(cpp-doc-xml
                ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile_xml
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                OUTPUT_DIRECTORY  ${CMAKE_BINARY_DIR}/docs/cpp
             COMMENT "Generating XML API documentation with Doxygen" VERBATIM
            )
            add_dependencies(doc cpp-doc-xml)
        endif()


    endif(DOXYGEN_FOUND)
endif(BUILD_DOCS)










