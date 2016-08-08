set(BOOST_COMPUTE_ROOT_DIR "" CACHE PATH "BOOST_COMPUTE root directory.")

find_path(BOOST_COMPUTE_INCLUDE_DIR boost/compute/version.hpp HINTS
    ${BOOST_COMPUTE_ROOT_DIR}/boost/compute/
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(BOOST_COMPUTE DEFAULT_MSG  BOOST_COMPUTE_INCLUDE_DIR   )

if(BOOST_COMPUTE_FOUND)

endif(BOOST_COMPUTE_FOUND)

mark_as_advanced(BOOST_COMPUTE_LIBRARY BOOST_COMPUTE_INCLUDE_DIR )
