#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>
#include "nifty/graph/simple_graph.hxx"
#include "nifty/graph/grid_region_adjacency_graph.hxx"

#include "export_undirected_graph_class_api.hxx"
#include "../converter.hxx"

namespace py = pybind11;


namespace nifty{
namespace graph{

    
    void exportGridRegionAdjacencyGraphRag3d(py::module & graphModule) {
    
        typedef UndirectedGraph<> Base;
        typedef Rag3d Graph;
        const auto clsName = std::string("Rag3d");
        auto graphCls = py::class_<Graph>(graphModule, clsName.c_str(),py::base<Base>());

        graphCls
            .def(py::init<>())
        ;

    }

    void exportGridRegionAdjacencyGraphRag3d2d(py::module & graphModule) {
    
        typedef UndirectedGraph<> Base;
        typedef Rag3d2d Graph;
        const auto clsName = std::string("Rag3d2d");
        auto graphCls = py::class_<Graph>(graphModule, clsName.c_str(),py::base<Base>());

        graphCls
            .def(py::init<>())
            .def("assignLabels",
                [](Graph & g, py::array_t<uint64_t> pyArray) {
                    NumpyArray<uint64_t> array(pyArray);
                    NIFTY_CHECK_OP(array.dimension(),==,3,"wrong dimensions");
                    g.assignLabels(array);
                }
            )
        ;

    }


    void exportGridRegionAdjacencyGraph(py::module & graphModule) {
        exportGridRegionAdjacencyGraphRag3d(graphModule);
        exportGridRegionAdjacencyGraphRag3d2d(graphModule);
    }

}
}
