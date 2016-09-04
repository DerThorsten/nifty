#pragma once
#ifndef NIFTY_PYTHON_GRAPH_MULTICUT_OBJECTIVE_NAME_HXX
#define NIFTY_PYTHON_GRAPH_MULTICUT_OBJECTIVE_NAME_HXX

#include "nifty/graph/optimization/multicut/multicut_objective.hxx"
#include "nifty/python/graph/graph_name.hxx"

namespace nifty{
namespace graph{



    template<class OBJECTIVE>
    struct MulticutObjectiveName;


    template<class GRAPH>
    using PyDefaultMulticutObjective = MulticutObjective<GRAPH, double>;

    template<class GRAPH>
    struct MulticutObjectiveName<PyDefaultMulticutObjective<GRAPH> >{
        static std::string name(){
            return std::string("MulticutObjective") + GraphName<GRAPH>::name();
        }
    };
    
}
}

#endif /* NIFTY_PYTHON_GRAPH_MULTICUT_OBJECTIVE_NAME_HXX */
