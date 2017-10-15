SET(MOD_LIBS ${NIFTY_LP_SOLVER_LIBS})
SET(MOD_SOURCES
        lifted_multicut.cxx
        lifted_multicut_objective.cxx
        lifted_multicut_visitor_base.cxx
        lifted_multicut_base.cxx
        lifted_multicut_factory.cxx
        lifted_multicut_greedy_additive.cxx
        lifted_multicut_kernighan_lin.cxx
        lifted_multicut_ilp.cxx
        #lifted_multicut_andres_kernighan_lin.cxx
        lifted_multicut_andres_greedy_additive.cxx
        fusion_move_based.cxx
        lifted_graph_features.cxx
)

if(WITH_LP_MP)
    SET(MOD_LIBS    ${MOD_LIBS} ${HDF5_LIBRARIES} )
    SET(MOD_SOURCES ${MOD_SOURCES} lifted_multicut_mp.cxx)
endif()

addPythonModule(
    NESTED_NAME 
        nifty/graph/opt/lifted_multicut  
    SOURCES
        ${MOD_SOURCES}
    LIBRRARIES
        ${MOD_LIBS}
)
