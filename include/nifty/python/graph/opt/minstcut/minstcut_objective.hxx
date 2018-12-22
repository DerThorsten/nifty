#pragma once

#include "nifty/graph/opt/minstcut/minstcut_objective.hxx"
#include "nifty/python/graph/graph_name.hxx"

namespace nifty{
namespace graph{
namespace opt{
namespace minstcut{


    template<class OBJECTIVE>
    struct MinstcutObjectiveName;


    template<class GRAPH>
    using PyDefaultMinstcutObjective = MinstcutObjective<GRAPH, double>;

    template<class GRAPH>
    struct MinstcutObjectiveName<PyDefaultMinstcutObjective<GRAPH> >{
        static std::string name(){
            return std::string("MinstcutObjective") + GraphName<GRAPH>::name();
        }
    };
} // namespace minstcut
} // namespace opt
}
}

