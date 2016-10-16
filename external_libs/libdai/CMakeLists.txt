
include(ExternalProject)
externalproject_add( libdai
    URL https://staff.fnwi.uva.nl/j.m.mooij/libDAI/libDAI-0.3.1.tar.gz
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${TMP_INSTALL_PREFIX}
               -D GMPXX_LIBRARY:PATH=${GMPXX_INCLUDE_DIR}
               -D GMPXX_LIBRARY:PATH=${GMPXX_LIBRARY}
               -D GMP_LIBRARY:PATH=${GMP_LIBRARY}
               -D GMP_INCLUDE_DIR:PATH=${GMP_INCLUDE_DIR}
)




#IF(CYGWIN)
#
#ELSEIF(WIN32)
#
#ELSEIF(APPLE)
#
#ELSEIF(UNIX)
#    ExternalProject_Add_Step(libdai e2
#        COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/libdai/libdai-prefix/src/libdai/Makefile.LINUX
#                                         ${PROJECT_BINARY_DIR}/libdai/libdai-prefix/src/libdai/Makefile.conf
#        DEPENDEES download
#        DEPENDERS configure
#    )
#
#ELSE()
#
#ENDIF()




ExternalProject_Add_Step(libdai copyCMakeFile
  COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/libdai/files/CMakeLists.txt ${PROJECT_BINARY_DIR}/libdai/libdai-prefix/src/libdai/CMakeLists.txt
  DEPENDEES patch
  DEPENDERS configure
) 





