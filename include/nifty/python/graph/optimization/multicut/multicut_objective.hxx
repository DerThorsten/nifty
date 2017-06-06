#pragma once

#include "nifty/graph/optimization/multicut/multicut_objective.hxx"
#include "nifty/python/graph/graph_name.hxx"

namespace nifty{
namespace graph{
namespace optimization{
namespace multicut{

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
 
} // namespace multicut 
} // namespace optimization   
}
}

