

# SET(MOD_LIBS "")
# SET(MOD_SOURCES
#     optimization.cxx
# )

# if(WITH_QPBO)
#     SET(MOD_LIBS ${MOD_LIBS})
# endif(WITH_QPBO)

addPythonModule(
    NESTED_NAME 
        nifty/graph/optimization  
    SOURCES
        optimization.cxx
    #LIBRRARIES
    #    ${MOD_LIBS}
)

add_subdirectory(mincut)
add_subdirectory(multicut)
add_subdirectory(lifted_multicut)