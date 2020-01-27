
addPythonModule(
    NESTED_NAME 
        nifty/transformation
    SOURCES
        transformation.cxx
        affine_transformation.cxx
    LIBRRARIES
        ${HDF5_LIBRARIES} 
        ${FASTFILTERS_LIBRARY}
        ${Z5_COMPRESSION_LIBRARIES}
        ${FILESYSTEM_LIBRARIES} 
        Threads::Threads
)
