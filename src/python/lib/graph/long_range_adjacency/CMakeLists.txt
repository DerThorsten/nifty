

addPythonModule(
    NESTED_NAME 
        nifty/graph/long_range_adjacency
    SOURCES
        adjacency.cxx 
        long_range_adjacency.cxx
        long_range_features.cxx
    LIBRRARIES
        ${HDF5_LIBRARIES}
        ${Boost_SYSTEM_LIBRARY}    
)
