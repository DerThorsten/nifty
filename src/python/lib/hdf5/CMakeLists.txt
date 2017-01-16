

addPythonModule(
    NESTED_NAME 
        nifty/hdf5  
    SOURCES
        hdf5.cxx
        hdf5_common.cxx
        hdf5_array.cxx
        hdf5_watershed.cxx
        hdf5_benchmark.cxx
    LIBRRARIES
        ${HDF5_LIBRARIES}
)