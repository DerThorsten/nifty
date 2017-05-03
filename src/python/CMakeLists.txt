add_subdirectory(lib)
add_subdirectory(test)



#################################
# INSTALL THE PYTHON MODULE
#################################

IF(NOT DEFINED PYTHON_MODULE_ISNTALL_DIR OR PYTHON_MODULE_ISNTALL_DIR MATCHES "^$")
        execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
                         "from distutils.sysconfig import *; print(get_python_lib(1))"
                          OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
        FILE(TO_CMAKE_PATH ${PYTHON_SITE_PACKAGES} PYTHON_MODULE_ISNTALL_DIR)
    ENDIF()
    SET(PYTHON_MODULE_ISNTALL_DIR ${PYTHON_MODULE_ISNTALL_DIR}
        CACHE PATH "where to install the VIGRA Python package" FORCE)
    # this is the install path relative to CMAKE_INSTALL_PREFIX,
    # use this in INSTALL() commands to get packaging right
    FILE(RELATIVE_PATH PYTHON_MODULE_ISNTALL_DIR ${CMAKE_INSTALL_PREFIX} ${PYTHON_MODULE_ISNTALL_DIR})






install(
    DIRECTORY ${CMAKE_BINARY_DIR}/python/nifty
    DESTINATION ${PYTHON_MODULE_ISNTALL_DIR}
    
    FILES_MATCHING
    PATTERN "*.py"
    PATTERN "*.so"
    PATTERN "*.pyd"
    PATTERN "CMakeFiles" EXCLUDE
)