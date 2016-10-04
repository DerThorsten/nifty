#pragma once
#ifndef NIFTY_PYTHON_GRAPH_LIFTED_MULTICUT_WEIGHTED_LIFTED_MULTICUT_OBJECTIVE_NAME_HXX
#define NIFTY_PYTHON_GRAPH_LIFTED_MULTICUT_WEIGHTED_LIFTED_MULTICUT_OBJECTIVE_NAME_HXX

#include "nifty/graph/optimization/lifted_multicut/weighted_lifted_multicut_objective.hxx"
#include "nifty/python/graph/graph_name.hxx"

namespace nifty{
namespace graph{
namespace lifted_multicut{


    template<class OBJECTIVE>
    struct LiftedMulticutObjectiveName;





    template<class GRAPH>
    struct LiftedMulticutObjectiveName<WeightedLiftedMulticutObjective<GRAPH, float> >{
        static std::string name(){
            return std::string("WeightedLiftedMulticutObjective") + GraphName<GRAPH>::name();
        }
    };

}
}
}

#endif /* NIFTY_PYTHON_GRAPH_LIFTED_MULTICUT_WEIGHTED_LIFTED_MULTICUT_OBJECTIVE_NAME_HXX */
