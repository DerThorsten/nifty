

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
    LIBRRARIES
    ${NIFTY_LP_SOLVER_LIBS} ${HDF5_LIBRARIES} ${FASTFILTERS_LIBRARY}
)
