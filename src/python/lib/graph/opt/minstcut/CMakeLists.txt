

SET(MOD_LIBS "")
SET(MOD_SOURCES
    minstcut.cxx
    minstcut_objective.cxx
    minstcut_visitor_base.cxx
    minstcut_base.cxx
    minstcut_factory.cxx
    # solvers
    #mincut_cc_fusion_move_based.cxx
)

#if(WITH_QPBO)
    # SET(MOD_LIBS    ${MOD_LIBS} external_qpbo)
    # SET(MOD_SOURCES ${MOD_SOURCES} 
    #     mincut_qpbo.cxx
    #     mincut_greedy_additive.cxx
    # )

#endif(WITH_QPBO)

addPythonModule(
    NESTED_NAME 
        nifty/graph/opt/minstcut  
    SOURCES
        ${MOD_SOURCES}
    LIBRRARIES
        ${MOD_LIBS}
)