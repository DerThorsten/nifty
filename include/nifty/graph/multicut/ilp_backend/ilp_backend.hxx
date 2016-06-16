#pragma once
#ifndef NIFTY_GRAPH_MULTICUT_ILP_BACKEND_HXX
#define NIFTY_GRAPH_MULTICUT_ILP_BACKEND_HXX

#include <limits>

namespace nifty {
namespace graph {
namespace ilp_backend{

    struct IlpBackendSettings{
        
        enum PreSolver {
            PRE_SOLVER_AUTO, 
            PRE_SOLVER_PRIMAL,
            PRE_SOLVER_DUAL, 
            PRE_SOLVER_NONE,
            PRE_SOLVER_DEFAULT
        };

        enum LPSolver {
            LP_SOLVER_PRIMAL_SIMPLEX, 
            LP_SOLVER_DUAL_SIMPLEX, 
            LP_SOLVER_BARRIER, 
            LP_SOLVER_SIFTING,
            LP_SOLVER_DEFAULT
        };

        double relativeGap{0.0};
        double absoluteGap{0.0};
        double cutUp{1.0e+75};
        int prePasses{-1};

        PreSolver preSolver{PRE_SOLVER_DEFAULT};
        LPSolver  lpSolver{LP_SOLVER_DEFAULT};

        size_t numberOfThreads{1};
        size_t verbosity{0};

    };  

} // namespace ilp_backend
} // namespace graph
} // namespace nifty

#endif
