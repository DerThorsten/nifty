if(HDF5_FOUND)
    add_executable(test_marray_hdf5 test_marray_hdf5.cxx )
    target_link_libraries(test_marray_hdf5 ${TEST_LIBS} ${HDF5_LIBRARIES})
    add_test(test_marray_hdf5 test_marray_hdf5)
endif()


add_executable(test_marray test_marray.cxx )
target_link_libraries(test_marray ${TEST_LIBS})
add_test(test_marray test_marray)