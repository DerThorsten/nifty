#-------------------------------------------------------------------------------------------------------------------
# src folder
#-------------------------------------------------------------------------------------------------------------------
#add_subdirectory(sandbox)
add_subdirectory(external_libs)

if(BUILD_CPP_TEST)
    add_subdirectory(test)
endif()
if(BUILD_NIFTY_PYTHON)
    add_subdirectory(python)
endif()

if(BUILD_CPP_EXAMPLES)
    add_subdirectory(examples)
endif()

#add_subdirectory(sandbox)
#add_subdirectory(benchmark)


