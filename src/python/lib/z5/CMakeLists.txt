
addPythonModule(
    NESTED_NAME
        nifty/z5
    SOURCES
        z5.cxx
        dataset_wrapper.cxx
        upsampling.cxx
    LIBRRARIES
        ${Z5_COMPRESSION_LIBRARIES} 
        ${Boost_FILESYSTEM_LIBRARY}    
        ${Boost_SYSTEM_LIBRARY}    
        pthread
)
