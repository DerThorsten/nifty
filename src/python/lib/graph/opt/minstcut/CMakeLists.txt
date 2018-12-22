

SET(MOD_LIBS "")
SET(MOD_SOURCES
    minstcut.cxx
    minstcut_objective.cxx
    minstcut_visitor_base.cxx
    minstcut_base.cxx
    minstcut_factory.cxx
    # solvers
    #mincut_cc_fusion_move_based.cxx
    #minstcut_maxflow.cxx
)


addPythonModule(
    NESTED_NAME 
        nifty/graph/opt/minstcut  
    SOURCES
        ${MOD_SOURCES}
    LIBRRARIES
        ${MOD_LIBS}
)
