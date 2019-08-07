
addPythonModule(
    NESTED_NAME 
        nifty/distributed
    SOURCES
        distributed.cxx
        graph_extraction.cxx
        distributed_graph.cxx
        mergeable_features.cxx
        distributed_utils.cxx
        lifted_utils.cxx
        morphology.cxx
        edge_morphology.cxx
    LIBRRARIES
        # ${FASTFILTERS_LIBRARY}
        ${Z5_COMPRESSION_LIBRARIES}
        ${FILESYSTEM_LIBRARIES}    
        Threads::Threads
)
