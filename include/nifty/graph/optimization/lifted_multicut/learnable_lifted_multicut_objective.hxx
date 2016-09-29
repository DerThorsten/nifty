#pragma once
#ifndef NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LEARNABLE_LIFTED_MULTICUT_OBJECTIVE_HXX
#define NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LEARNABLE_LIFTED_MULTICUT_OBJECTIVE_HXX


#include "nifty/graph/optimization/lifted_multicut/lifted_multicut_objective.hxx"
#include "nifty/structured_learning/instances/weighted_edge.hxx"

namespace nifty{
namespace graph{
namespace lifted_multicut{






    template<class GRAPH, class WEIGHT_TYPE>   
    class LearnableLiftedMulticutObjective :  
        public LiftedMulticutObjective<GRAPH, WEIGHT_TYPE>
    {   
    public:


    private:

    };

} // namespace lifted_multicut
} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LEARNABLE_LIFTED_MULTICUT_OBJECTIVE_HXX
