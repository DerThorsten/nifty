#include <pybind11/pybind11.h>

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pyvectorize.hpp"

#include <iostream>


namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);


namespace nifty{
namespace graph{


    void exportUndirectedListGraph(py::module &);
    void exportUndirectedGridGraph(py::module &);
    void exportUndirectedLongRangeGridGraph(py::module &);
    void exportEdgeContractionGraphUndirectedGraph(py::module & );
    void exportShortestPathDijkstra(py::module &);
    void exportConnectedComponents(py::module &);
    void exportEdgeWeightedWatersheds(py::module &);
    void exportNodeWeightedWatersheds(py::module &);
}
}


PYBIND11_MODULE(_graph, module) {

    xt::import_numpy();
    
    py::options options;
    options.disable_function_signatures();
    
    module.doc() = "graph submodule of nifty";

    using namespace nifty::graph;

        

    exportUndirectedListGraph(module);
    exportUndirectedGridGraph(module);
    exportUndirectedLongRangeGridGraph(module);
    exportEdgeContractionGraphUndirectedGraph(module);
    exportShortestPathDijkstra(module);
    exportConnectedComponents(module);
    exportEdgeWeightedWatersheds(module);
    exportNodeWeightedWatersheds(module);

}

