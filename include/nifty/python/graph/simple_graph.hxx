#pragma once
#ifndef NIFTY_PYTHON_GRAPH_SIMPLE_GRAPH_HXX
#define NIFTY_PYTHON_GRAPH_SIMPLE_GRAPH_HXX

#include <string>

#include "nifty/python/graph/graph_name.hxx"
#include "nifty/graph/simple_graph.hxx"


namespace nifty{
namespace graph{

    typedef UndirectedGraph<> PyUndirectedGraph;

    template<>
    struct GraphName<PyUndirectedGraph>{
        static std::string name(){
            return std::string("UndirectedGraph");
        }
    };

}
}

#endif // NIFTY_PYTHON_GRAPH_SIMPLE_GRAPH_HXX
