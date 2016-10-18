#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

#include "nifty/python/converter.hxx"

#include "nifty/graph/undirected_list_graph.hxx"

namespace py = pybind11;

namespace nifty{
namespace structured_learning{

    void exportStructMaxMargin(py::module & groundTruthModule);
    void exportStructMaxMarginOracleLmc(py::module & groundTruthModule);
}
}








PYBIND11_PLUGIN(_structured_learning) {
    py::module structuredLearningModule("_structured_learning", "structured learning submodule of nifty python bindings");

    using namespace nifty::structured_learning;

        

    exportStructMaxMargin(structuredLearningModule);
    exportStructMaxMarginOracleLmc(structuredLearningModule);
        
    return structuredLearningModule.ptr();
}
