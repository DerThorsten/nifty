
include(ExternalProject)
externalproject_add( qpbo
    URL         http://pub.ist.ac.at/~vnk/software/QPBO-v1.3.src.tar.gz
    PATCH_COMMAND patch 
        -s -d ${EXTERNAL_LIBS_BINARY_DIR}/qpbo/qpbo-prefix/src/qpbo -p1 < ${CMAKE_CURRENT_SOURCE_DIR}/files/QPBO-v1.3.patch -N -r -
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${TMP_INSTALL_PREFIX}
)



ExternalProject_Add_Step(qpbo copyCMakeFile
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/files/CMakeLists.txt ${EXTERNAL_LIBS_BINARY_DIR}/qpbo/qpbo-prefix/src/qpbo/CMakeLists.txt
  DEPENDEES patch
  DEPENDERS configure
) 


set_target_properties(qpbo PROPERTIES EXCLUDE_FROM_ALL TRUE)

add_dependencies(external_libs qpbo)