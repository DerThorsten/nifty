#pragma once
#ifndef NIFTY_PYTHON_GRAPH_LIFTED_MULTICUT_LIFTED_MULTICUT_OBJECTIVE_NAME_HXX
#define NIFTY_PYTHON_GRAPH_LIFTED_MULTICUT_LIFTED_MULTICUT_OBJECTIVE_NAME_HXX

#include "nifty/graph/lifted_multicut/lifted_multicut_objective.hxx"
#include "nifty/python/graph/graph_name.hxx"

namespace nifty{
namespace graph{
namespace lifted_multicut{


    template<class OBJECTIVE>
    struct LiftedMulticutObjectiveName;


    template<class GRAPH>
    using PyDefaultMulticutObjective = LiftedMulticutObjective<GRAPH, double>;


    template<class GRAPH>
    struct LiftedMulticutObjectiveName<PyDefaultMulticutObjective<GRAPH> >{
        static std::string name(){
            return std::string("LiftedMulticutObjective") + GraphName<GRAPH>::name();
        }
    };

}
}
}

#endif /* NIFTY_PYTHON_GRAPH_LIFTED_MULTICUT_LIFTED_MULTICUT_OBJECTIVE_NAME_HXX */
