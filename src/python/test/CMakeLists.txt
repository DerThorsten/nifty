if(BUILD_PYTHON_TEST)

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

    
endif()


include(PythonLibTest)

if(NOSETESTS_PATH)

    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/python_test)
    

    add_custom_target(python-test)


    add_python_test_target(python-test-core

        hdf5/test_hdf5.py
        tools/test_blocking.py
        graph/agglo/test_agglo.py
        graph/mincut/test_mincut_objective.py
        graph/mincut/test_mincut_solvers.py
        graph/multicut/test_multicut_solvers.py
        graph/lifted_multicut/test_lifted_multicut_objective.py
        graph/lifted_multicut/test_lifted_multicut_solvers.py
        graph/lifted_multicut/test_lifted_graph_features.py
        cgp/test_cgp.py
    )
    add_dependencies(python-test-core python-module)
    add_dependencies(python-test python-test-core)

endif()




