
include(ExternalProject)
externalproject_add( ad3
    #GIT_REPOSITORY git@github.com:andre-martins/AD3.git  
    GIT_REPOSITORY https://github.com/andre-martins/AD3.git
    PATCH_COMMAND patch 
        -s -d ${PROJECT_BINARY_DIR}/ad3/ad3-prefix/src/ad3 -p1 < ${PROJECT_SOURCE_DIR}/ad3/files/AD3.patch -N -r -
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${TMP_INSTALL_PREFIX}
)

ExternalProject_Add_Step(ad3 copyCMakeFile
  COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/ad3/files/CMakeLists.txt ${PROJECT_BINARY_DIR}/ad3/ad3-prefix/src/ad3/CMakeLists.txt
  DEPENDEES patch
  DEPENDERS configure
) 


