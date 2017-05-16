add_executable(test_histogram test_histogram.cxx )
target_link_libraries(test_histogram ${TEST_LIBS})
add_test(test_histogram test_histogram)
