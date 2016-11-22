
addPythonModule(
    NESTED_NAME 
        nifty/pipelines  
    SOURCES
        pipelines.cxx
    LIBRRARIES
        ${HDF5_LIBRARIES}
)

add_subdirectory(neuro_seg)
