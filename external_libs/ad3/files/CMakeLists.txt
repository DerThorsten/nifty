cmake_minimum_required(VERSION 2.8)

set(AD3_SRC_FILES
  ad3/FactorGraph.cpp
  ad3/GenericFactor.cpp
  ad3/Factor.cpp
  ad3/Utils.cpp
)
   

include_directories( ad3 )
include_directories( Eigen )


ADD_LIBRARY(inferno-external-ad3   ${AD3_SRC_FILES})
ADD_LIBRARY(inferno-external-ad3-shared   SHARED ${AD3_SRC_FILES} )

install (TARGETS inferno-external-ad3 DESTINATION lib/inferno_externals/)
install (TARGETS inferno-external-ad3-shared DESTINATION lib/inferno_externals/)


install(DIRECTORY ad3
    DESTINATION include/inferno_externals
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN ".svn" EXCLUDE
)

install(DIRECTORY Eigen
    DESTINATION include/inferno_externals
    PATTERN ".svn" EXCLUDE
)

