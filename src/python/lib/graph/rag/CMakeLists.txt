
addPythonModule(
    NESTED_NAME 
        nifty/graph/rag  
    SOURCES
        rag.cxx
        grid_rag.cxx
        graph_accumulator.cxx
        project_to_pixels.cxx
        accumulate.cxx
    LIBRRARIES
        ${NIFTY_LP_SOLVER_LIBS} ${HDF5_LIBRARIES}
)