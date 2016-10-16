
include(ExternalProject)

if( CMAKE_MAJOR_VERSION GREATER 2 OR CMAKE_MINOR_VERSION GREATER 8 OR CMAKE_PATCH_VERSION GREATER 10)
  externalproject_add( trws
    URL http://ftp.research.microsoft.com/downloads/dad6c31e-2c04-471f-b724-ded18bf70fe3/TRW_S-v1.3.zip
    PATCH_COMMAND patch 
    -s -d ${PROJECT_BINARY_DIR}/trws/trws-prefix/src/trws -p1 < ${PROJECT_SOURCE_DIR}/trws/files/TRWS-v1.3.patch -N -r -
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${TMP_INSTALL_PREFIX}
    )
else()
  message(STATUS "Use manual download/unzip for TRWS since CMAKE version might be buggy - this manual script might not work for all platforms ... ")
  externalproject_add( trws
    DOWNLOAD_COMMAND  wget -q http://ftp.research.microsoft.com/downloads/dad6c31e-2c04-471f-b724-ded18bf70fe3/TRW_S-v1.3.zip -P /tmp/
    PATCH_COMMAND patch 
    -s -d ${PROJECT_BINARY_DIR}/trws/trws-prefix/src/trws -p1 < ${PROJECT_SOURCE_DIR}/trws/files/TRWS-v1.3.patch -N -r -
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${TMP_INSTALL_PREFIX}
    )
  
  
  ExternalProject_Add_Step(trws UnzipFile
    COMMAND unzip -juqn /tmp/TRW_S-v1.3.zip -d ${PROJECT_BINARY_DIR}/trws/trws-prefix/src/trws
    DEPENDEES download
    DEPENDERS patch
    ) 
endif()

ExternalProject_Add_Step(trws copyCMakeFile
  COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/trws/files/CMakeLists.txt ${PROJECT_BINARY_DIR}/trws/trws-prefix/src/trws/CMakeLists.txt
  DEPENDEES patch
  DEPENDERS configure
) 


