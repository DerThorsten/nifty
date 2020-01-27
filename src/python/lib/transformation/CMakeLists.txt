# TODO only link libraries if we build with z5
addPythonModule(
    NESTED_NAME 
        nifty/transformation
    SOURCES
        transformation.cxx
        affine_transformation.cxx
    LIBRRARIES
        ${Z5_COMPRESSION_LIBRARIES}
        ${FILESYSTEM_LIBRARIES}
        Threads::Threads
)
