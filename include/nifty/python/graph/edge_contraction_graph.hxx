#pragma once

#include <string>
#include <set>

#include "nifty/python/graph/graph_name.hxx"
#include "nifty/graph/edge_contraction_graph.hxx"


namespace nifty{
namespace graph{

    template<class BASE_GRAPH>
    using PyContractionGraph = EdgeContractionGraphWithSets<
        BASE_GRAPH, 
        FlexibleCallback, 
        std::set<uint64_t> 
    >;


    template<class BASE_GRAPH>
    struct GraphName< PyContractionGraph<BASE_GRAPH> >{
        static std::string name(){
            return std::string("EdgeContractionGraph") + GraphName<BASE_GRAPH>::name();
        }

        static std::string moduleName(){
            return std::string("nifty.graph");
        }

        static std::string usageExample(){
            return std::string(
                "import nifty\n"
            );
        }
    };

}
}

