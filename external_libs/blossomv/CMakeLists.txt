include(ExternalProject)
externalproject_add( blossomv
    URL http://pub.ist.ac.at/~vnk/software/blossom5-v2.05.src.tar.gz
    #PATCH_COMMAND patch 
    #    -s -d ${PROJECT_BINARY_DIR}/blossomv/blossomv-prefix/src/blossomv -p1 < ${PROJECT_SOURCE_DIR}/blossomv/files/blossomv.patch -N -r -
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${TMP_INSTALL_PREFIX}
)

ExternalProject_Add_Step(blossomv copyCMakeFile
  COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/blossomv/files/CMakeLists.txt ${PROJECT_BINARY_DIR}/blossomv/blossomv-prefix/src/blossomv/CMakeLists.txt
  DEPENDEES patch
  DEPENDERS configure
) 

