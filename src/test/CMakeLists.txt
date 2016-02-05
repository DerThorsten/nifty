
#-------------------------------------------------------------------------------------------------------------------
# unittest
#-------------------------------------------------------------------------------------------------------------------
find_package(Boost COMPONENTS  unit_test_framework REQUIRED)
ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK) 

SET(TEST_LIBS
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
)

add_subdirectory(test_graph)
add_subdirectory(test_array)
add_subdirectory(test_marray)

if(WITH_BOOST_COMPUTE)
    add_subdirectory(test_compute)
endif()
