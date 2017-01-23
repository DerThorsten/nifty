
addPythonModule(
    NESTED_NAME 
        nifty/pipelines/ilastik_backend
    SOURCES
        ilastik_backend.cxx
        interactive_pixel_classification.cxx
    LIBRRARIES
        ${HDF5_LIBRARIES}
)

