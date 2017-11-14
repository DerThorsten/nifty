
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
        ${HDF5_LIBRARIES}   
        #${NIFTY_LP_SOLVER_LIBS} ${HDF5_LIBRARIES}
)

target_link_libraries(_rag PUBLIC ${HDF5_LIBRARIES}   )