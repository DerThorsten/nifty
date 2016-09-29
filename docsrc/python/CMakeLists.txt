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
        # Sphinx cache with pickled ReST documents
        set(SPHINX_CACHE_DIR "${CMAKE_BINARY_DIR}/docs/python/_doctrees")
        # HTML output directory
        set(SPHINX_HTML_DIR "${CMAKE_BINARY_DIR}/docs/python/html")
        configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/conf.py.in"
            "${SPHINX_BUILD_DIR}/conf.py"
            @ONLY)
        add_custom_target(python-doc 
            ${SPHINX_EXECUTABLE}
            -q -b html
            -v -v -v 
            -j 9
            -c "${SPHINX_BUILD_DIR}"
            -d "${SPHINX_CACHE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}"
            "${SPHINX_HTML_DIR}"
            COMMENT "Building HTML documentation with Sphinx")
        install(DIRECTORY ${SPHINX_HTML_DIR}
            DESTINATION share/doc/python-opengm
        )
        add_dependencies(python-doc _nifty )
        add_dependencies(python-doc _graph )
        add_dependencies(python-doc _rag )

        add_dependencies(doc python-doc)
endif()



