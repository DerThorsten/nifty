
include(ExternalProject)
externalproject_add( maxflow
    URL         http://pub.ist.ac.at/~vnk/software/maxflow-v3.02.src.tar.gz
    PATCH_COMMAND patch 
        -s -d ${PROJECT_BINARY_DIR}/maxflow/maxflow-prefix/src/maxflow -p1 < ${PROJECT_SOURCE_DIR}/maxflow/files/maxflow-v3.02.patch -N -r -
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${TMP_INSTALL_PREFIX}
)

ExternalProject_Add_Step(maxflow copyCMakeFile
  COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/maxflow/files/CMakeLists.txt ${PROJECT_BINARY_DIR}/maxflow/maxflow-prefix/src/maxflow/CMakeLists.txt
  DEPENDEES patch
  DEPENDERS configure
) 


