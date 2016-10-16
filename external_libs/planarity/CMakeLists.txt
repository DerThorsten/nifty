include(ExternalProject)
externalproject_add( planarity
    URL http://jgaa.info/accepted/2004/BoyerMyrvold2004.8.3/planarity.zip
    PATCH_COMMAND patch 
        -s -d ${PROJECT_BINARY_DIR}/planarity/planarity-prefix/src/planarity -p1 < ${PROJECT_SOURCE_DIR}/planarity/files/planarity.patch -N -r -
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${TMP_INSTALL_PREFIX}
)

ExternalProject_Add_Step(planarity copyCMakeFile
  COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/planarity/files/CMakeLists.txt ${PROJECT_BINARY_DIR}/planarity/planarity-prefix/src/planarity/CMakeLists.txt
  DEPENDEES patch
  DEPENDERS configure
) 

