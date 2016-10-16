cmake_minimum_required(VERSION 2.8)




add_definitions(-DMRFLABELVALUE=int)
add_definitions(-DMRFENERGYVALUE=double)
add_definitions(-DMRFCOSTVALUE=double)

set(bitness 32)
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  MESSAGE (STATUS "Detected a 64 Bit Machine")
  set(bitness 64)
  add_definitions(-DUSE_64_BIT_PTR_CAST)
else()
  MESSAGE (STATUS "Detected a 32 Bit Machine")
endif()



    SET(MRF_SRC_FILES
        mrf.cpp
        ICM.cpp
        GCoptimization.cpp
        graph.cpp
        maxflow.cpp
        MaxProdBP.cpp
        LinkedBlockList.cpp
        regions-maxprod.cpp
        TRW-S.cpp
        BP-S.cpp
    )

#include_directories( ad3 )
#include_directories( Eigen )


ADD_LIBRARY(inferno-external-mrf   ${MRF_SRC_FILES})
ADD_LIBRARY(inferno-external-mrf-shared   SHARED ${MRF_SRC_FILES} )

install (TARGETS inferno-external-mrf DESTINATION lib/inferno_externals/)
install (TARGETS inferno-external-mrf-shared DESTINATION lib/inferno_externals/)


install(DIRECTORY "."
    DESTINATION include/inferno_externals/mrf
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN ".svn" EXCLUDE
)
#
#install(DIRECTORY Eigen
#    DESTINATION include/inferno_externals
#    PATTERN ".svn" EXCLUDE
#)

