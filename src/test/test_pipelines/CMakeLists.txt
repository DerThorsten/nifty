find_package(VIGRA)

add_executable(test_ilastik_pixel_classification test_ilastik_pixel_classification.cxx )
target_link_libraries(test_ilastik_pixel_classification ${TEST_LIBS} ${TBB_LIBRARIES} ${FASTFILTERS_LIBRARY}
        ${HDF5_LIBRARIES} ${VIGRA_IMPEX_LIBRARY})
add_test(test_ilastik_pixel_classification test_ilastik_pixel_classification)
