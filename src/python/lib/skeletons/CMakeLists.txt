
addPythonModule(
    NESTED_NAME 
        nifty/skeletons
    SOURCES
        skeletons.cxx
        evaluation.cxx
    LIBRRARIES
        ${Z5_COMPRESSION_LIBRARIES}
        ${FILESYSTEM_LIBRARIES}    
        ${Boost_SERIALIZATION_LIBRARY}    
        Threads::Threads
)
