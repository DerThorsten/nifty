if(WITH_QPBO)
    add_definitions(-DWITH_QPBO)
    add_library(external_qpbo SHARED 
        "${CMAKE_SOURCE_DIR}/externals/qpbo/QPBO.cpp"
        "${CMAKE_SOURCE_DIR}/externals/qpbo/QPBO_extra.cpp"
        "${CMAKE_SOURCE_DIR}/externals/qpbo/QPBO_maxflow.cpp"
        "${CMAKE_SOURCE_DIR}/externals/qpbo/QPBO_postprocessing.cpp"
    )
endif(WITH_QPBO)