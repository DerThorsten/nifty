
addPythonModule(
    NESTED_NAME 
        nifty/skeletons
    SOURCES
        skeletons.cxx
        evaluation.cxx
    LIBRRARIES
        ${Z5_COMPRESSION_LIBRARIES}
        ${Boost_FILESYSTEM_LIBRARY}    
        ${Boost_SYSTEM_LIBRARY}    
        ${Boost_SERIALIZATION_LIBRARY}    
        Threads::Threads
)
