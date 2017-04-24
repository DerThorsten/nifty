#-------------------------------------------------------------------------------------------------------------------
# Documentation
#-------------------------------------------------------------------------------------------------------------------
if(BUILD_DOCS)
    add_custom_target(doc )
    add_subdirectory(cpp)
    add_subdirectory(python)

    install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/doc"
    DESTINATION share/doc/nifty
    PATTERN ".git" EXCLUDE PATTERN ".cmake" EXCLUDE
    )

endif()


