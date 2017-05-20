#-------------------------------------------------------------------------------------------------------------------
# Documentation
#-------------------------------------------------------------------------------------------------------------------
if(BUILD_DOCS)
    add_custom_target(doc )
    add_subdirectory(cpp)
    add_subdirectory(python)

    install(DIRECTORY "${CMAKE_BINARY_DIR}/docs"
    DESTINATION share/doc/nifty
    PATTERN ".git" EXCLUDE PATTERN ".cmake" EXCLUDE
    )

endif()


