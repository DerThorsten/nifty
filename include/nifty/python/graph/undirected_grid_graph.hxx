#pragma once
#include <string>

#include "nifty/python/graph/graph_name.hxx"
#include "nifty/graph/undirected_grid_graph.hxx"


namespace nifty{
namespace graph{



    template<
        size_t DIM,
        bool SIMPLE_NH
    >
    struct GraphName<UndirectedGridGraph<DIM, SIMPLE_NH>>{
        static std::string name(){
            return std::string("UndirectedGridGraph") + 
                std::to_string(DIM) +
                std::string("D") + 
                (SIMPLE_NH ? 
                    std::string("SimpleNh") : 
                    std::string("ExtendedNh"));
                
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
