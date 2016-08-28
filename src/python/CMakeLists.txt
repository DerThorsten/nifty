add_subdirectory(lib)


# add python tests here
include(PythonLibTest)

if(NOSETESTS_PATH)
   add_python_test_target(pythonTest   _nifty 

       test/test_python.py
       test/test_undirected_list_graph.py
       test/test_multicut_inference.py
       test/test_rag.py
       test/test_rag_accumulate.py
       test/test_agglo.py
       test/test_hdf5.py

       # graph
       ## lifted multicut
       test/graph/lifted_multicut/test_lifted_multicut_objective.py
    )
endif()
