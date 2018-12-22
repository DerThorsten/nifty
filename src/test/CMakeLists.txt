
#-------------------------------------------------------------------------------------------------------------------
# unittest
#-------------------------------------------------------------------------------------------------------------------

SET(TEST_LIBS "")

add_subdirectory(test_graph)
add_subdirectory(test_array)
add_subdirectory(test_features)
add_subdirectory(test_histogram)

add_executable(test_blocking test_blocking.cxx )
target_link_libraries(test_blocking ${TEST_LIBS})
add_test(test_blocking test_blocking)
