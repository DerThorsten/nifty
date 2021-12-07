addPythonModule(
    NESTED_NAME 
        nifty/graph  
    SOURCES
        graph.cxx
        undirected_list_graph.cxx
        undirected_grid_graph.cxx
        edge_weighted_watersheds.cxx
        node_weighted_watersheds.cxx
        edge_contraction_graph_undirected_graph.cxx
        export_shortest_path_dijkstra.cxx
        connected_components.cxx
        label_propagation.cxx
        accumulate_long_range_affinities.cxx
        LIBRRARIES
        ${HDF5_LIBRARIES} 
)


# submodules
add_subdirectory(agglo)
add_subdirectory(rag)
add_subdirectory(opt)
add_subdirectory(long_range_adjacency)
