
include(ExternalProject)
externalproject_add( daoopt
   # GIT_REPOSITORY git@github.com:joergkappes/daoopt.git
    GIT_REPOSITORY https://github.com/joergkappes/daoopt.git
    #PATCH_COMMAND patch 
    #    -s -d ${PROJECT_BINARY_DIR}/daoopt/daoopt-prefix/src/daoopt -p1 < ${PROJECT_SOURCE_DIR}/daoopt/files/daoopt.patch -N -r -
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${TMP_INSTALL_PREFIX}
)

ExternalProject_Add_Step(daoopt copyCMakeFile
  COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/daoopt/files/CMakeLists.txt ${PROJECT_BINARY_DIR}/daoopt/daoopt-prefix/src/daoopt/CMakeLists.txt
  DEPENDEES patch
  DEPENDERS configure
) 


