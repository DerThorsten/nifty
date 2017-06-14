if(BUILD_NIFTY_PYTHON AND BUILD_PYTHON_DOCS)

    find_package(SPHINX REQUIRED)

        if(NOT DEFINED SPHINX_THEME)
            set(SPHINX_THEME default)
        endif()
        if(NOT DEFINED SPHINX_THEME_DIR)
            set(SPHINX_THEME_DIR)
        endif()
        # configured documentation tools and intermediate build results
        set(SPHINX_BUILD_DIR "${CMAKE_BINARY_DIR}/docs/python/_build")

        file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/docs/python/_build/_static")

        # Sphinx cache with pickled ReST documents
        set(SPHINX_CACHE_DIR "${CMAKE_BINARY_DIR}/docs/python/_doctrees")
        # HTML output directory
        set(SPHINX_HTML_DIR "${CMAKE_BINARY_DIR}/docs/python/html")

        configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/conf.py.in"
            "${SPHINX_BUILD_DIR}/conf.py"
            @ONLY)

        configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/_templates/template_module.rst"
            "${SPHINX_BUILD_DIR}/_templates/template_module.rst"
            @ONLY)

        configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/../common/refs.bib"
            "${CMAKE_CURRENT_SOURCE_DIR}/refs.bib"
            @ONLY)
        configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/../common/refs.bib"
            "${SPHINX_BUILD_DIR}/refs.bib"
            @ONLY)


        add_custom_target(python-doc 
            ${SPHINX_EXECUTABLE}
            -q -b html -E
            -v -v -v 
            -j 9
            -c "${SPHINX_BUILD_DIR}"
            -d "${SPHINX_CACHE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}"
            "${SPHINX_HTML_DIR}"
            COMMENT "Building HTML documentation with Sphinx")
        #install(DIRECTORY ${SPHINX_HTML_DIR}
        #    DESTINATION share/doc/nifty_python
        #)

        # if there is a cpp doc 
        if(BUILD_DOCS)
            if(DOXYGEN_FOUND)
                    add_dependencies(python-doc cpp-doc-xml)
            endif(DOXYGEN_FOUND)
        endif(BUILD_DOCS)

        add_dependencies(python-doc python-module )
        add_dependencies(doc python-doc)


endif()



