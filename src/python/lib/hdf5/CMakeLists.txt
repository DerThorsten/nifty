# NOTE: this is probably a hack, but without this the windows build
# does not work. Longer term, the right solution is probably to use
# the hdf5 exported targets provided by the installed CMake files.
add_definitions(-DH5_BUILT_AS_DYNAMIC_LIB=1)

addPythonModule(
    NESTED_NAME
        nifty/hdf5
    SOURCES
        hdf5.cxx
        hdf5_common.cxx
        hdf5_array.cxx
        #hdf5_watershed.cxx
        hdf5_benchmark.cxx
    LIBRRARIES
        ${HDF5_LIBRARIES}
)
