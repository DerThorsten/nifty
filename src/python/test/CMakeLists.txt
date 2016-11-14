
include(PythonLibTest)

if(NOSETESTS_PATH)

    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/python_test)


    add_custom_target(python-test)



    add_python_test_target(python-test-tools
        # unsorted
        tools/test_blocking.py
        #test_python.py
        #test_undirected_list_graph.py
        #test_multicut_inference.py
        #test_rag.py
        #test_rag_accumulate.py
        #test_hdf5.py
        ## graph
        ### agglo
        #graph/agglo/test_agglo.py
        ### lifted multicut
        #graph/lifted_multicut/test_lifted_multicut_objective.py
        #graph/lifted_multicut/test_lifted_multicut_solvers.py
        #graph/lifted_multicut/test_lifted_graph_features.py
    )

    add_python_test_target(python-test-graph
        ## graph
        ### agglo
        graph/agglo/test_agglo.py
        ### lifted multicut
        graph/lifted_multicut/test_lifted_multicut_objective.py
        graph/lifted_multicut/test_lifted_multicut_solvers.py
        graph/lifted_multicut/test_lifted_graph_features.py
    )

    add_dependencies(python-test python-test-tools)
    add_dependencies(python-test python-test-graph)


endif()
