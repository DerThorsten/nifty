
#-------------------------------------------------------------------------------------------------------------------
# unittest
#-------------------------------------------------------------------------------------------------------------------
find_package(Boost COMPONENTS  unit_test_framework REQUIRED)
ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK) 
if(BUILD_CPP_TEST_WITH_GLIBCXX_CXX11_ABI)
else()
    ADD_DEFINITIONS(-D_GLIBCXX_USE_CXX11_ABI=0)
endif()
SET(TEST_LIBS
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
)

add_subdirectory(test_graph)
add_subdirectory(test_array)
add_subdirectory(test_features)
add_subdirectory(test_histogram)



add_executable(test_marray test_marray.cxx )
target_link_libraries(test_marray ${TEST_LIBS})
add_test(test_marray test_marray)


add_executable(test_blocking test_blocking.cxx )
target_link_libraries(test_blocking ${TEST_LIBS})
add_test(test_blocking test_blocking)
