

include(ExternalProject)



SET(FASTPD_URL "" CACHE STRING "URL for downloading FastPD (Registration required at http://www.csd.uoc.gr/~komod/FastPD/)")
SET(FASTPD_ENERGY_VALUE "double" CACHE STRING "selected energy type for FASTPD library")
add_definitions(-DFASTPDENERGYVALUE=${FASTPD_ENERGY_VALUE})
SET(FASTPD_LABEL_VALUE "size_t" CACHE STRING "selected label type for FASTPD library")
add_definitions(-DFASTPDLABELVALUE=${FASTPD_LABEL_VALUE})



externalproject_add( fastpd
    URL ${FASTPD_URL}
    PATCH_COMMAND patch 
        -s -d ${PROJECT_BINARY_DIR}/fastpd/fastpd-prefix/src/fastpd/ -p6 < ${PROJECT_SOURCE_DIR}/fastpd/files/FastPD.patch -N -r -
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${TMP_INSTALL_PREFIX} 
)



ExternalProject_Add_Step(fastpd copyCMakeFile
  COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/fastpd/files/CMakeLists.txt ${PROJECT_BINARY_DIR}/fastpd/fastpd-prefix/src/fastpd/CMakeLists.txt
  DEPENDEES patch
  DEPENDERS configure
) 


