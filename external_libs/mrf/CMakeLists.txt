

include(ExternalProject)



if( CMAKE_MAJOR_VERSION GREATER 2 OR CMAKE_MINOR_VERSION GREATER 8 OR CMAKE_PATCH_VERSION GREATER 10)
  externalproject_add( mrftrws
    URL http://ftp.research.microsoft.com/downloads/dad6c31e-2c04-471f-b724-ded18bf70fe3/TRW_S-v1.3.zip
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    )
else()
  externalproject_add( mrftrws
    DOWNLOAD_COMMAND  wget -q http://ftp.research.microsoft.com/downloads/dad6c31e-2c04-471f-b724-ded18bf70fe3/TRW_S-v1.3.zip -P /tmp/
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    )  
  ExternalProject_Add_Step( mrftrws UnzipFile
    COMMAND unzip -juqn /tmp/TRW_S-v1.3.zip -d ${PROJECT_BINARY_DIR}/mrf/mrftrws-prefix/src/mrftrws/
    DEPENDEES download
    DEPENDERS patch
    ) 

endif()



externalproject_add( mrf
    URL http://vision.middlebury.edu/MRF/code/MRF2.1.zip
    PATCH_COMMAND patch 
        -s -d ${PROJECT_BINARY_DIR}/mrf/mrf-prefix/src/mrf -p1 < ${PROJECT_SOURCE_DIR}/mrf/files/MRF2.1.patch -N -r -
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${TMP_INSTALL_PREFIX} 
)

add_dependencies(mrf mrftrws)


ExternalProject_Add_Step(mrf copytypeTruncatedQuadratic2DFile
  COMMAND ${CMAKE_COMMAND} -E copy  ${PROJECT_BINARY_DIR}/mrf/mrftrws-prefix/src/mrftrws/typeTruncatedQuadratic2D.h
  										   ${PROJECT_BINARY_DIR}/mrf/mrf-prefix/src/mrf/typeTruncatedQuadratic2D.h
  DEPENDEES patch
  DEPENDERS configure
) 


ExternalProject_Add_Step(mrf copyCMakeFile
  COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/mrf/files/CMakeLists.txt ${PROJECT_BINARY_DIR}/mrf/mrf-prefix/src/mrf/CMakeLists.txt
  DEPENDEES patch
  DEPENDERS configure
) 


