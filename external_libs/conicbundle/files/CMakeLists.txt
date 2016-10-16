cmake_minimum_required(VERSION 2.8)

SET(CONICBUNDLE_SRC_FILES
${CMAKE_CURRENT_SOURCE_DIR}/CBsources/lmaxproblem.cxx
${CMAKE_CURRENT_SOURCE_DIR}/CBsources/funproblem.cxx
${CMAKE_CURRENT_SOURCE_DIR}/CBsources/MatSDPfun.cxx
${CMAKE_CURRENT_SOURCE_DIR}/CBsources/coneproblem.cxx
${CMAKE_CURRENT_SOURCE_DIR}/CBsources/sumproblem.cxx
${CMAKE_CURRENT_SOURCE_DIR}/CBsources/MatFCBSolver.cxx
${CMAKE_CURRENT_SOURCE_DIR}/CBsources/Nbundle.cxx
${CMAKE_CURRENT_SOURCE_DIR}/CBsources/socproblem.cxx
${CMAKE_CURRENT_SOURCE_DIR}/CBsources/MatNBSolver.cxx
${CMAKE_CURRENT_SOURCE_DIR}/CBsources/qp_sdpblock.cxx
${CMAKE_CURRENT_SOURCE_DIR}/CBsources/bundle.cxx
${CMAKE_CURRENT_SOURCE_DIR}/CBsources/qp_solver.cxx
${CMAKE_CURRENT_SOURCE_DIR}/CBsources/lowranktrSMscaling.cxx
${CMAKE_CURRENT_SOURCE_DIR}/CBsources/lowrankSMscaling.cxx
${CMAKE_CURRENT_SOURCE_DIR}/CBsources/MatCBSolver.cxx
${CMAKE_CURRENT_SOURCE_DIR}/CBsources/lowranktrscaling.cxx
${CMAKE_CURRENT_SOURCE_DIR}/CBsources/lowrankscaling.cxx
${CMAKE_CURRENT_SOURCE_DIR}/CBsources/CB_CSolver.cxx
${CMAKE_CURRENT_SOURCE_DIR}/CBsources/CBSolver.cxx
${CMAKE_CURRENT_SOURCE_DIR}/CBsources/Nfunproblem.cxx
${CMAKE_CURRENT_SOURCE_DIR}/CBsources/diagtrscaling.cxx
${CMAKE_CURRENT_SOURCE_DIR}/CBsources/MatConcavefun.cxx
${CMAKE_CURRENT_SOURCE_DIR}/CBsources/diagscaling.cxx
${CMAKE_CURRENT_SOURCE_DIR}/CBsources/fulltrscaling.cxx
${CMAKE_CURRENT_SOURCE_DIR}/CBsources/bigmat.cxx
${CMAKE_CURRENT_SOURCE_DIR}/CBsources/fullscaling.cxx
${CMAKE_CURRENT_SOURCE_DIR}/CBsources/MatLPBCfun.cxx
${CMAKE_CURRENT_SOURCE_DIR}/CBsources/idscaling.cxx
${CMAKE_CURRENT_SOURCE_DIR}/CBsources/MatConefun.cxx
${CMAKE_CURRENT_SOURCE_DIR}/CBsources/Nsumproblem.cxx
${CMAKE_CURRENT_SOURCE_DIR}/CBsources/problem.cxx
${CMAKE_CURRENT_SOURCE_DIR}/CBsources/hkweight.cxx
${CMAKE_CURRENT_SOURCE_DIR}/CBsources/MatLPfun.cxx
${CMAKE_CURRENT_SOURCE_DIR}/CBsources/coeffmat.cxx
${CMAKE_CURRENT_SOURCE_DIR}/CBsources/CFunction.cxx
${CMAKE_CURRENT_SOURCE_DIR}/CBsources/MatSOCfun.cxx    
${CMAKE_CURRENT_SOURCE_DIR}/Matrix/sparsmat.cxx
${CMAKE_CURRENT_SOURCE_DIR}/Matrix/lanczpol.cxx
${CMAKE_CURRENT_SOURCE_DIR}/Matrix/sparssym.cxx
${CMAKE_CURRENT_SOURCE_DIR}/Matrix/matrix.cxx
${CMAKE_CURRENT_SOURCE_DIR}/Matrix/symmat.cxx
${CMAKE_CURRENT_SOURCE_DIR}/Matrix/indexmat.cxx
${CMAKE_CURRENT_SOURCE_DIR}/Matrix/eigval.cxx
${CMAKE_CURRENT_SOURCE_DIR}/Matrix/qr.cxx
${CMAKE_CURRENT_SOURCE_DIR}/Matrix/memarray.cxx
${CMAKE_CURRENT_SOURCE_DIR}/Matrix/chol.cxx
${CMAKE_CURRENT_SOURCE_DIR}/Matrix/nnls.cxx
${CMAKE_CURRENT_SOURCE_DIR}/Matrix/ldl.cxx
${CMAKE_CURRENT_SOURCE_DIR}/Matrix/trisolve.cxx
)


include_directories( include )
include_directories( Matrix )
include_directories( Tools )
include_directories( CBsources )

ADD_LIBRARY(inferno-external-conicbundle   ${CONICBUNDLE_SRC_FILES})
ADD_LIBRARY(inferno-external-conicbundle-shared   SHARED ${CONICBUNDLE_SRC_FILES} )

install (TARGETS inferno-external-conicbundle DESTINATION lib/inferno_externals/)
install (TARGETS inferno-external-conicbundle-shared DESTINATION lib/inferno_externals/)


install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/CBsources
    DESTINATION include/inferno_externals/conicbundle/
    FILES_MATCHING
    PATTERN "*.hxx"
    PATTERN ".svn" EXCLUDE
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Tools
    DESTINATION include/inferno_externals/conicbundle/
    FILES_MATCHING
    PATTERN "*.hxx"
    PATTERN ".svn" EXCLUDE
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Matrix
    DESTINATION include/inferno_externals/conicbundle/
    FILES_MATCHING
    PATTERN "*.hxx"
    PATTERN ".svn" EXCLUDE
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include
    DESTINATION include/inferno_externals/conicbundle
    FILES_MATCHING
    PATTERN "*.hxx"
    PATTERN ".svn" EXCLUDE
)
