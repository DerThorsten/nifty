
SET(MOD_LIBS ${NIFTY_LP_SOLVER_LIBS})
SET(MOD_SOURCES
        ho_multicut.cxx
        ho_multicut_objective.cxx
        # ho_multicut_visitor_base.cxx
        # ho_multicut_base.cxx
        # ho_multicut_factory.cxx
        # ho_multicut_ilp.cxx
        # ho_multicut_decomposer.cxx
        # ho_multicut_greedy_additive.cxx
        # ho_multicut_andres.cxx
        # fusion_move_based.cxx
        # cc_fusion_move_based.cxx
        # perturb_and_map.cxx
        # chained_solvers.cxx
        # cgc.cxx
        # kernighan_lin.cxx
)

# if(WITH_LP_MP)
#     SET(MOD_LIBS    ${MOD_LIBS} ${HDF5_LIBRARIES} )#llgl)
#     SET(MOD_SOURCES ${MOD_SOURCES} multicut_mp.cxx)
# endif(WITH_LP_MP)

addPythonModule(
    NESTED_NAME 
        nifty/graph/opt/ho_multicut  
    SOURCES
        ${MOD_SOURCES}
    LIBRRARIES
        ${MOD_LIBS}
)

target_link_libraries(_ho_multicut PUBLIC ${MOD_LIBS} )