cmake_minimum_required(VERSION 2.8)


SET(LIBDAI_SRC_FILES
${CMAKE_CURRENT_SOURCE_DIR}/src/alldai.cpp  
${CMAKE_CURRENT_SOURCE_DIR}/src/bipgraph.cpp  
${CMAKE_CURRENT_SOURCE_DIR}/src/bp_dual.cpp  
${CMAKE_CURRENT_SOURCE_DIR}/src/clustergraph.cpp 
${CMAKE_CURRENT_SOURCE_DIR}/src/daialg.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/emalg.cpp    
${CMAKE_CURRENT_SOURCE_DIR}/src/exactinf.cpp   
${CMAKE_CURRENT_SOURCE_DIR}/src/factor.cpp     
${CMAKE_CURRENT_SOURCE_DIR}/src/fbp.cpp   
${CMAKE_CURRENT_SOURCE_DIR}/src/graph.cpp 
${CMAKE_CURRENT_SOURCE_DIR}/src/io.cpp   
${CMAKE_CURRENT_SOURCE_DIR}/src/lc.cpp  
${CMAKE_CURRENT_SOURCE_DIR}/src/mf.cpp 
${CMAKE_CURRENT_SOURCE_DIR}/src/properties.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/treeep.cpp 
${CMAKE_CURRENT_SOURCE_DIR}/src/util.cpp  
${CMAKE_CURRENT_SOURCE_DIR}/src/weightedgraph.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/bbp.cpp    
${CMAKE_CURRENT_SOURCE_DIR}/src/bp.cpp       
${CMAKE_CURRENT_SOURCE_DIR}/src/cbp.cpp    
${CMAKE_CURRENT_SOURCE_DIR}/src/dag.cpp    
${CMAKE_CURRENT_SOURCE_DIR}/src/decmap.cpp  
${CMAKE_CURRENT_SOURCE_DIR}/src/evidence.cpp 
${CMAKE_CURRENT_SOURCE_DIR}/src/exceptions.cpp 
${CMAKE_CURRENT_SOURCE_DIR}/src/factorgraph.cpp  
${CMAKE_CURRENT_SOURCE_DIR}/src/gibbs.cpp 
${CMAKE_CURRENT_SOURCE_DIR}/src/hak.cpp  
${CMAKE_CURRENT_SOURCE_DIR}/src/jtree.cpp  
${CMAKE_CURRENT_SOURCE_DIR}/src/mr.cpp  
${CMAKE_CURRENT_SOURCE_DIR}/src/regiongraph.cpp  
${CMAKE_CURRENT_SOURCE_DIR}/src/trwbp.cpp   
${CMAKE_CURRENT_SOURCE_DIR}/src/varset.cpp
)


add_definitions(-DDAI_WITH_BP)
add_definitions(-DDAI_WITH_FBP)
add_definitions(-DDAI_WITH_TRWBP)
add_definitions(-DDAI_WITH_MF)
add_definitions(-DDAI_WITH_HAK)
add_definitions(-DDAI_WITH_LC)
add_definitions(-DDAI_WITH_TREEEP)
add_definitions(-DDAI_WITH_JTREE)
add_definitions(-DDAI_WITH_MR)
add_definitions(-DDAI_WITH_GIBBS)
add_definitions(-DDAI_WITH_CBP)
add_definitions(-DDAI_WITH_DECMAP)



include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/include )
include_directories( ${GMP_INCLUDE_DIR} )
include_directories( ${GMPXX_INCLUDE_DIR} )

ADD_LIBRARY(inferno-external-libdai   ${LIBDAI_SRC_FILES})
ADD_LIBRARY(inferno-external-libdai-shared   SHARED ${LIBDAI_SRC_FILES} )

install (TARGETS inferno-external-libdai DESTINATION lib/inferno_externals/)
install (TARGETS inferno-external-libdai-shared DESTINATION lib/inferno_externals/)


install(DIRECTORY include/dai
    DESTINATION include/inferno_externals/
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN ".svn" EXCLUDE
)
