set(PYBIND_ELEVEN_ROOT_DIR "" CACHE PATH "PYBIND_ELEVEN root directory.")

find_path(PYBIND_ELEVEN_INCLUDE_DIR pybind_11/pybind11.h HINTS
    ${PYBIND_ELEVEN_ROOT_DIR}/pybind11/include
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PYBIND_ELEVEN DEFAULT_MSG  PYBIND_ELEVEN_INCLUDE_DIR   )

if(PYBIND_ELEVEN_FOUND)

endif(PYBIND_ELEVEN_FOUND)

mark_as_advanced(PYBIND_ELEVEN_LIBRARY PYBIND_ELEVEN_INCLUDE_DIR )
