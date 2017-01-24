
addPythonModule(
    NESTED_NAME 
        nifty/pipelines/ilastik_backend
    SOURCES
        ilastik_backend.cxx
        interactive_pixel_classification.cxx
        pixel_classification_prototype.cxx
    LIBRRARIES
        ${TBB_LIBRARIES} ${FASTFILTER_LIBRARY}
        ${HDF5_LIBRARIES}

)

