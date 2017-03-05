add_executable(simple_rag_test simple_rag_test.cxx )
target_link_libraries(simple_rag_test ${TEST_LIBS})
target_link_libraries(simple_rag_test ${HDF5_LIBRARIES})
add_test(simple_rag_test simple_rag_test)
