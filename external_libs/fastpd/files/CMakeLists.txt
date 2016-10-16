cmake_minimum_required(VERSION 2.8)


SET(FASTPD_SRC_FILES
    ${FASTPD_PATCHEDSRCDIR}/graph.cpp
    ${FASTPD_PATCHEDSRCDIR}/LinkedBlockList.cpp
    ${FASTPD_PATCHEDSRCDIR}/maxflow.cpp
)


ADD_LIBRARY(inferno-external-fastpd   ${FASTPD_SRC_FILES})
ADD_LIBRARY(inferno-external-fastpd-shared   SHARED ${FASTPD_SRC_FILES} )

install (TARGETS inferno-external-fastpd DESTINATION lib/inferno_externals/)
install (TARGETS inferno-external-fastpd-shared DESTINATION lib/inferno_externals/)


install(DIRECTORY "."
    DESTINATION include/inferno_externals/fastpd
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN ".svn" EXCLUDE
)
