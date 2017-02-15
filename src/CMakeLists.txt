#-------------------------------------------------------------------------------------------------------------------
# src folder
#-------------------------------------------------------------------------------------------------------------------
add_subdirectory(test)

if(BUILD_NIFTY_PYTHON)
    add_subdirectory(python)
endif()

if(BUILD_CPP_EXAMPLES)
    add_subdirectory(examples)
endif()


add_subdirectory(benchmark)
