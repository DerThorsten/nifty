add_executable(test_compute test_compute.cxx )
message(STATUS ${OpenCL_LIBRARIES})
target_link_libraries(test_compute ${TEST_LIBS}  ${OpenCL_LIBRARIES}) 
add_test(test_compute test_compute)

