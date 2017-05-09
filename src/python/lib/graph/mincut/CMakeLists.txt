

SET(MOD_LIBS ${NIFTY_LP_SOLVER_LIBS})
SET(MOD_SOURCES
        mincut.cxx
        mincut_objective.cxx
        mincut_visitor_base.cxx
        mincut_base.cxx
        mincut_factory.cxx
)

if(WITH_QPBO)
    SET(MOD_LIBS    ${MOD_LIBS} external_qpbo)
    SET(MOD_SOURCES ${MOD_SOURCES} mincut_qpbo.cxx)
endif(WITH_QPBO)

addPythonModule(
    NESTED_NAME 
        nifty/graph/mincut  
    SOURCES
        ${MOD_SOURCES}
    LIBRRARIES
        ${MOD_LIBS}
)