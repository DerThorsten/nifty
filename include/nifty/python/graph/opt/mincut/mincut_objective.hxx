#pragma once

#include "nifty/graph/opt/mincut/mincut_objective.hxx"
#include "nifty/python/graph/graph_name.hxx"

namespace nifty{
namespace graph{
namespace opt{
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
} // namespace opt   
}
}

