#pragma once

#include <string>

#include "nifty/python/graph/graph_name.hxx"
#include "nifty/graph/undirected_list_graph.hxx"


namespace nifty{
namespace graph{

    typedef UndirectedGraph<> PyUndirectedGraph;

    template<>
    struct GraphName<PyUndirectedGraph>{
        static std::string name(){
            return std::string("UndirectedGraph");
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

