

include(ExternalProject)





externalproject_add( mplp
    URL http://cs.nyu.edu/~dsontag/code/mplp_ver2.tgz
    PATCH_COMMAND patch 
        -s -d ${PROJECT_BINARY_DIR}/mplp/mplp-prefix/src/mplp -p1 < ${PROJECT_SOURCE_DIR}/mplp/files/mplp_ver2.patch -N -r -
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${TMP_INSTALL_PREFIX} 
)



ExternalProject_Add_Step(mplp copyCMakeFile
  COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/mplp/files/CMakeLists.txt ${PROJECT_BINARY_DIR}/mplp/mplp-prefix/src/mplp/CMakeLists.txt
  DEPENDEES patch
  DEPENDERS configure
) 


