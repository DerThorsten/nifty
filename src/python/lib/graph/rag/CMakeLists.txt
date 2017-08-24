

addPythonModule(
    NESTED_NAME 
        nifty/graph/rag  
    SOURCES
        rag.cxx
        grid_rag.cxx
        grid_rag_stacked.cxx
        graph_accumulator.cxx
        project_to_pixels.cxx
        accumulate.cxx
        accumulate_stacked.cxx
        accumulate_filters_stacked.cxx
        accumulate_flat.cxx
        grid_rag_coordinates.cxx
    LIBRRARIES
        ${HDF5_LIBRARIES} ${FASTFILTERS_LIBRARY}# ${NIFTY_LP_SOLVER_LIBS}
)
