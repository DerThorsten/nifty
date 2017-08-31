#pragma once

#include "nifty/graph/opt/lifted_multicut/lifted_multicut_objective.hxx"
#include "nifty/python/graph/graph_name.hxx"

namespace nifty{
namespace graph{
namespace opt{
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
} // namespace opt
}
}

