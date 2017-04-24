#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);


namespace nifty{
namespace graph{


    void exportUndirectedListGraph(py::module &);
    void exportEdgeContractionGraphUndirectedGraph(py::module & );
    void exportShortestPathDijkstra(py::module &);



}
}


PYBIND11_PLUGIN(_graph) {
    py::module graphModule("_graph", "graph submodule of nifty");

    using namespace nifty::graph;

        

    exportUndirectedListGraph(graphModule);
    exportEdgeContractionGraphUndirectedGraph(graphModule);
    exportShortestPathDijkstra(graphModule);


        
    return graphModule.ptr();
}

