#include <pybind11/pybind11.h>
#include <iostream>

#include "nifty/graph/simple_graph.hxx"
#include "nifty/graph/grid_region_adjacency_graph.hxx"

namespace py = pybind11;


namespace nifty{
namespace graph{


    void exportUndirectedGraph(py::module &);
    void exportGridRegionAdjacencyGraph(py::module &);

    void initSubmoduleGraph(py::module &niftyModule) {
        auto graphModule = niftyModule.def_submodule("graph","graph submodule");

        exportUndirectedGraph(graphModule);
        exportGridRegionAdjacencyGraph(graphModule);
    }

}
}
