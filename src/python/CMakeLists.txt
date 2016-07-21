add_subdirectory(lib)


# add python tests here
include(PythonLibTest)
add_python_test_target(pythonTest _nifty test/test_python.py)