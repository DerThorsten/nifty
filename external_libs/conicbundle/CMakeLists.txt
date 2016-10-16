
include(ExternalProject)
externalproject_add( conicbundle
    URL http://www-user.tu-chemnitz.de/~helmberg/ConicBundle/CB_v0.3.11.tgz
    PATCH_COMMAND patch 
        -s -d ${PROJECT_BINARY_DIR}/conicbundle/conicbundle-prefix/src/conicbundle -p1 < ${PROJECT_SOURCE_DIR}/conicbundle/files/ConicBundle-v0.3.11.patch -N -r -
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${TMP_INSTALL_PREFIX}
)


ExternalProject_Add_Step(conicbundle copyConfigFile
  COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/conicbundle/files/CBconfig.hxx ${PROJECT_BINARY_DIR}/conicbundle/conicbundle-prefix/src/conicbundle/include/CBconfig.hxx
  DEPENDEES patch
  DEPENDERS configure
) 


ExternalProject_Add_Step(conicbundle copyCMakeFile
  COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/conicbundle/files/CMakeLists.txt ${PROJECT_BINARY_DIR}/conicbundle/conicbundle-prefix/src/conicbundle/CMakeLists.txt
  DEPENDEES patch
  DEPENDERS configure
) 


