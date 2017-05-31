#pragma once
#ifndef NIFTY_PYTHON_GRAPH_MULTICUT_OBJECTIVE_NAME_HXX
#define NIFTY_PYTHON_GRAPH_MULTICUT_OBJECTIVE_NAME_HXX

#include "nifty/graph/optimization/mincut/mincut_objective.hxx"
#include "nifty/python/graph/graph_name.hxx"

namespace nifty{
namespace graph{
namespace optimization{
namespace mincut{


    template<class OBJECTIVE>
    struct MincutObjectiveName;


    template<class GRAPH>
    using PyDefaultMincutObjective = MincutObjective<GRAPH, double>;

    template<class GRAPH>
    struct MincutObjectiveName<PyDefaultMincutObjective<GRAPH> >{
        static std::string name(){
            return std::string("MincutObjective") + GraphName<GRAPH>::name();
        }
    };
} // namespace mincut
} // namespace optimization   
}
}

#endif /* NIFTY_PYTHON_GRAPH_MULTICUT_OBJECTIVE_NAME_HXX */
