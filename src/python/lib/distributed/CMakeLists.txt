
addPythonModule(
    NESTED_NAME 
        nifty/distributed
    SOURCES
        distributed.cxx
        graph_extraction.cxx
        distributed_graph.cxx
        mergeable_features.cxx
    LIBRRARIES
        # ${FASTFILTERS_LIBRARY}
        ${Z5_COMPRESSION_LIBRARIES}
        ${Boost_FILESYSTEM_LIBRARY}    
        ${Boost_SYSTEM_LIBRARY}    
        pthread
)
