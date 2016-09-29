#pragma once
#ifndef NIFTY_PYTHON_GRAPH_LIFTED_MULTICUT_LEARNABLE_LIFTED_MULTICUT_OBJECTIVE_NAME_HXX
#define NIFTY_PYTHON_GRAPH_LIFTED_MULTICUT_LEARNABLE_LIFTED_MULTICUT_OBJECTIVE_NAME_HXX

#include "nifty/graph/optimization/lifted_multicut/learnable_lifted_multicut_objective.hxx"
#include "nifty/python/graph/graph_name.hxx"

namespace nifty{
namespace graph{
namespace lifted_multicut{


    template<class OBJECTIVE>
    struct LiftedMulticutObjectiveName;





    template<class GRAPH>
    struct LiftedMulticutObjectiveName<LearnableLiftedMulticutObjective<GRAPH, float> >{
        static std::string name(){
            return std::string("LearnableLiftedMulticutObjective") + GraphName<GRAPH>::name();
        }
    };

}
}
}

#endif /* NIFTY_PYTHON_GRAPH_LIFTED_MULTICUT_LEARNABLE_LIFTED_MULTICUT_OBJECTIVE_NAME_HXX */
