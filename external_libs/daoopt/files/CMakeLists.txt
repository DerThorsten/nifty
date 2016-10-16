cmake_minimum_required(VERSION 2.8)

   set(DAOOPT_ZLIB_SRC_FILES
      ${CMAKE_CURRENT_SOURCE_DIR}/lib/zlib/adler32.c
      ${CMAKE_CURRENT_SOURCE_DIR}/lib/zlib/compress.c
      ${CMAKE_CURRENT_SOURCE_DIR}/lib/zlib/crc32.c
      ${CMAKE_CURRENT_SOURCE_DIR}/lib/zlib/deflate.c
      ${CMAKE_CURRENT_SOURCE_DIR}/lib/zlib/gzclose.c
      ${CMAKE_CURRENT_SOURCE_DIR}/lib/zlib/gzlib.c
      ${CMAKE_CURRENT_SOURCE_DIR}/lib/zlib/gzread.c
      ${CMAKE_CURRENT_SOURCE_DIR}/lib/zlib/gzwrite.c
      ${CMAKE_CURRENT_SOURCE_DIR}/lib/zlib/infback.c
      ${CMAKE_CURRENT_SOURCE_DIR}/lib/zlib/inffast.c
      ${CMAKE_CURRENT_SOURCE_DIR}/lib/zlib/inflate.c
      ${CMAKE_CURRENT_SOURCE_DIR}/lib/zlib/inftrees.c
      ${CMAKE_CURRENT_SOURCE_DIR}/lib/zlib/trees.c
      ${CMAKE_CURRENT_SOURCE_DIR}/lib/zlib/uncompr.c
      ${CMAKE_CURRENT_SOURCE_DIR}/lib/zlib/zutil.c


      ${CMAKE_CURRENT_SOURCE_DIR}/lib/sls4mpe/AssignmentManager.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/lib/sls4mpe/Bucket.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/lib/sls4mpe/fheap.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/lib/sls4mpe/main_algo.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/lib/sls4mpe/MiniBucketElimination.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/lib/sls4mpe/my_set.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/lib/sls4mpe/ProbabilityTable.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/lib/sls4mpe/ProblemReader.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/lib/sls4mpe/random_numbers.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/lib/sls4mpe/timer.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/lib/sls4mpe/Variable.cpp

      ${CMAKE_CURRENT_SOURCE_DIR}/source/BestFirst.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/source/BoundPropagator.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/source/BoundPropagatorMaster.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/source/BranchAndBound.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/source/BranchAndBoundMaster.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/source/BranchAndBoundRotate.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/source/BranchAndBoundSampler.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/source/CacheTable.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/source/Function.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/source/Graph.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/source/LearningEngine.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/source/LimitedDiscrepancy.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/source/Main.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/source/MiniBucket.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/source/MiniBucketElim.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/source/ParallelManager.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/source/Problem.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/source/ProgramOptions.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/source/Pseudotree.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/source/Random.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/source/Search.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/source/SearchMaster.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/source/SearchNode.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/source/SigHandler.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/source/SLSWrapper.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/source/Statistics.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/source/SubproblemCondor.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/source/SubprobStats.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/source/utils.cpp
   )
   
   
   
   include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
                          ${CMAKE_CURRENT_SOURCE_DIR}/lib
      )

 # To select the two master modes
option(DAOOPT_WORKER "Force worker binary" OFF)
option(DAOOPT_MASTER_STATIC "Force static master binary" OFF)
option(DAOOPT_MASTER_DYNAMIC "Force dynamic master binary" OFF)
  
if(DAOOPT_WORKER)
   add_definitions(-DNOTHREADS)
else()
   if(DAOOPT_MASTER_STATIC)
      add_definitions(-DPARALLEL_STATIC)
   endif()
   if(DAOOPT_MASTER_DYNAMIC AND NOT DAOOPT_MASTER_STATIC)
      add_definitions(-DPARALLEL_DYNAMIC)
      find_package(Threads)  # includes pthread
      find_package( Boost REQUIRED COMPONENTS system )
   endif()
endif()


# To enable static linking of the final daoopt binary
option(DAOOPT_LINK_STATIC "Link binary statically" OFF)

# Add defines for DAOOPT
if(WIN32)
   add_definitions(-DWINDOWS)
else()
   add_definitions(-DLINUX)
endif()

#include_directories( daoopt )
#include_directories( Eigen )


ADD_LIBRARY(inferno-external-daoopt   ${DAOOPT_ZLIB_SRC_FILES})
ADD_LIBRARY(inferno-external-daoopt-shared   SHARED ${DAOOPT_ZLIB_SRC_FILES} )

install (TARGETS inferno-external-daoopt DESTINATION lib/inferno_externals/)
install (TARGETS inferno-external-daoopt-shared DESTINATION lib/inferno_externals/)


install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
   DESTINATION include/inferno_externals/daoopt
   FILES_MATCHING
   PATTERN "*.h"
   PATTERN ".svn" EXCLUDE
)

#
#install(DIRECTORY Eigen
#    DESTINATION include/inferno_externals
#    PATTERN ".svn" EXCLUDE
#)
#
