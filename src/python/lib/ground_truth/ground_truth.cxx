#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

#include "nifty/python/converter.hxx"

#include "nifty/graph/undirected_list_graph.hxx"

namespace py = pybind11;

namespace nifty{
namespace ground_truth{


    void exportOverlap(py::module & groundTruthModule);
    void exportPartitionComparison(py::module & groundTruthModule);
    void exportSegToLiftedEdges(py::module & groundTruthModule);
    void exportSegToEdges(py::module & groundTruthModule);
}
}








PYBIND11_MODULE(_ground_truth, groundTruthModule) {

    py::options options;
    options.disable_function_signatures();
    
    groundTruthModule.doc() = ("ground truth submodule of nifty python bindings");

    using namespace nifty::ground_truth;

        

    exportOverlap(groundTruthModule);
    exportPartitionComparison(groundTruthModule);
    exportSegToLiftedEdges(groundTruthModule);
    exportSegToEdges(groundTruthModule);

}
