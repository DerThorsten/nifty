cmake_minimum_required(VERSION 2.8)


SET(MPLP_SRC_FILES
    matrix.cpp
    mplp_alg.cpp
    muldim_arr.cpp
    read_model_file.cpp
)

#include_directories( ad3 )
#include_directories( Eigen )

ADD_LIBRARY(inferno-external-mplp   ${MPLP_SRC_FILES})
ADD_LIBRARY(inferno-external-mplp-shared   SHARED ${MPLP_SRC_FILES} )

install (TARGETS inferno-external-mplp DESTINATION lib/inferno_externals/)
install (TARGETS inferno-external-mplp-shared DESTINATION lib/inferno_externals/)


install(DIRECTORY "."
    DESTINATION include/inferno_externals/mplp
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN ".svn" EXCLUDE
)
