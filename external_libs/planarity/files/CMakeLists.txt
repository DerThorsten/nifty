cmake_minimum_required(VERSION 2.8)


SET(SRC_FILES
graphEmbed.c 
graphIO.c  
graphIsolator.c  
graphNonplanar.c  
graphPreprocess.c  
graphStructure.c  
graphTests.c  
listcoll.c  
planarity.c  
stack.c
)


ADD_LIBRARY(inferno-external-planarity   ${SRC_FILES})
ADD_LIBRARY(inferno-external-planarity-shared   SHARED ${SRC_FILES} )

install (TARGETS inferno-external-planarity DESTINATION lib/inferno_externals/)
install (TARGETS inferno-external-planarity-shared DESTINATION lib/inferno_externals/)


install(DIRECTORY "."
    DESTINATION include/inferno_externals/planarity
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN ".svn" EXCLUDE
)
