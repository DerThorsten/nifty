#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);


namespace nifty{
namespace graph{


    void exportUndirectedListGraph(py::module &);
    void exportUndirectedGridGraph(py::module &);
    void exportEdgeContractionGraphUndirectedGraph(py::module & );
    void exportShortestPathDijkstra(py::module &);
    void exportConnectedComponents(py::module &);

}
}


PYBIND11_PLUGIN(_graph) {
    py::module module("_graph", "graph submodule of nifty");

    using namespace nifty::graph;

        

    exportUndirectedListGraph(module);
    exportUndirectedGridGraph(module);
    exportEdgeContractionGraphUndirectedGraph(module);
    exportShortestPathDijkstra(module);
    exportConnectedComponents(module);
        
    return module.ptr();
}

