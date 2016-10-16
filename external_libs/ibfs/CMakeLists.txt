
include(ExternalProject)
externalproject_add( ibfs
    URL http://www.cs.tau.ac.il/~sagihed/ibfs/download/ibfs.tar
    #$PATCH_COMMAND patch 
    #    -s -d ${PROJECT_BINARY_DIR}/ibfs/ibfs-prefix/src/ibfs -p1 < ${PROJECT_SOURCE_DIR}/ibfs/files/ibfs.patch -N -r -
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${TMP_INSTALL_PREFIX}
)

ExternalProject_Add_Step(ibfs copyCMakeFile
  COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/ibfs/files/CMakeLists.txt ${PROJECT_BINARY_DIR}/ibfs/ibfs-prefix/src/ibfs/CMakeLists.txt
  DEPENDEES patch
  DEPENDERS configure
) 


