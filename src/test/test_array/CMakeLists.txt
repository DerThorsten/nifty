add_executable(test_array test_array.cxx )
target_link_libraries(test_array ${TEST_LIBS})
add_test(test_array test_array)

