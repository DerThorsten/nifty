#pragma once

#include "nifty/graph/optimization/lifted_multicut/learnable_lifted_multicut_objective.hxx"
#include "nifty/python/graph/graph_name.hxx"

namespace nifty{
namespace graph{
namespace optimization{
namespace lifted_multicut{


    template<class OBJECTIVE>
    struct LiftedMulticutObjectiveName;





    template<class GRAPH, class T>
    struct LiftedMulticutObjectiveName<LearnableLiftedMulticutObjective<GRAPH, T> >{
        static std::string name(){
            return std::string("LearnableLiftedMulticutObjective") + GraphName<GRAPH>::name();
        }
    };

}
} // namespace optimization
}
}

