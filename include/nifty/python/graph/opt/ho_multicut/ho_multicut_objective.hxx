#pragma once

#include "nifty/graph/opt/ho_multicut/ho_multicut_objective.hxx"
#include "nifty/python/graph/graph_name.hxx"

namespace nifty{
namespace graph{
namespace opt{
namespace ho_multicut{

    template<class OBJECTIVE>
    struct HoMulticutObjectiveName;


    template<class GRAPH>
    using PyDefaultHoMulticutObjective = HoMulticutObjective<GRAPH, double>;

    template<class GRAPH>
    struct HoMulticutObjectiveName<PyDefaultHoMulticutObjective<GRAPH> >{
        static std::string name(){
            return std::string("HoMulticutObjective") + GraphName<GRAPH>::name();
        }
    };
 
} // namespace ho_multicut 
} // namespace opt   
}
}

