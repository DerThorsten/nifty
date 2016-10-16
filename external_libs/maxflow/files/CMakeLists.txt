cmake_minimum_required(VERSION 2.8)


 SET(MAXFLOW_SRC_FILES
        graph.cpp
        maxflow.cpp
)

ADD_LIBRARY(inferno-external-maxflow ${MAXFLOW_SRC_FILES})
ADD_LIBRARY(inferno-external-maxflow-shared SHARED ${MAXFLOW_SRC_FILES})

install (TARGETS inferno-external-maxflow DESTINATION lib/inferno_externals/)
install (TARGETS inferno-external-maxflow-shared DESTINATION lib/inferno_externals/)

install(DIRECTORY "."
    DESTINATION include/inferno_externals/maxflow
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN ".svn" EXCLUDE
)
