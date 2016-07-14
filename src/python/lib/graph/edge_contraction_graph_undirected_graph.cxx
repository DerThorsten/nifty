#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>

#include "nifty/python/graph/simple_graph.hxx"

#include "export_edge_contraction_graph.hxx"
#include "nifty/python/converter.hxx"

namespace py = pybind11;


namespace nifty{
namespace graph{



    void exportEdgeContractionGraphUndirectedGraph(py::module & graphModule) {

        exportEdgeContractionGraphCallback(graphModule);
        
        typedef UndirectedGraph<> BaseGraphType;
        const auto baseGraphClsName = std::string("UndirectedGraph");
        auto cls = exportEdgeContractionGraph<BaseGraphType>(graphModule, baseGraphClsName);

    }

}
}
