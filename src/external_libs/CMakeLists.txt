add_subdirectory(qpbo)
#if(WITH_LP_MP)
#    add_subdirectory(lp_mp)
#    # For now we don't need to link against lingeling
#    #add_subdirectory(lingeling)
#endif(WITH_LP_MP)
