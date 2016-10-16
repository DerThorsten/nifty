cmake_minimum_required(VERSION 2.8)


SET(SRC_FILES
#example.cpp  
misc.cpp 
PMduals.cpp  
PMexpand.cpp  
PMinit.cpp  
PMinterface.cpp  
PMmain.cpp  
PMrepair.cpp  
PMshrink.cpp
)


ADD_LIBRARY(inferno-external-blossomv   ${SRC_FILES})
ADD_LIBRARY(inferno-external-blossomv-shared   SHARED ${SRC_FILES} )

install (TARGETS inferno-external-blossomv DESTINATION lib/inferno_externals/)
install (TARGETS inferno-external-blossomv-shared DESTINATION lib/inferno_externals/)


install(DIRECTORY "."
    DESTINATION include/inferno_externals/blossomv
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN ".svn" EXCLUDE
)
