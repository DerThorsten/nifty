add_subdirectory(lib)


# add python tests here
include(PythonLibTest)

if(NOSETESTS_PATH)

    

    add_python_test_target(pythonTest   _nifty 
        test/test_python.py
        test/test_rag.py
    )



endif()
