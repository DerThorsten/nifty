addPythonModule(
    NESTED_NAME 
        nifty/tools 
    SOURCES
        tools.cxx
        make_dense.cxx
        blocking.cxx
        take.cxx
        unique.cxx
        nodes_to_blocks.cxx
    LIBRRARIES
        ${HDF5_LIBRARIES}
)

