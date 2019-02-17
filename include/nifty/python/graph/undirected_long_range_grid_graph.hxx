#pragma once
#include <string>

#include "nifty/python/graph/graph_name.hxx"
#include "nifty/graph/undirected_long_range_grid_graph.hxx"


namespace nifty{
namespace graph{



    template<
        std::size_t DIM
    >
    struct GraphName<UndirectedLongRangeGridGraph<DIM>>{
        static std::string name(){
            return std::string("UndirectedLongRangeGridGraph") + 
                std::to_string(DIM) +
                std::string("D");
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
