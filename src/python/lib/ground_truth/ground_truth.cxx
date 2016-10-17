#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

#include "nifty/python/converter.hxx"

#include "nifty/graph/undirected_list_graph.hxx"

namespace py = pybind11;

namespace nifty{
namespace ground_truth{

    void exportPostProcessCarvingNeuroGroundTruth(py::module & groundTruthModule);
    void exportOverlap(py::module & groundTruthModule);
    void exportPartitionComparison(py::module & groundTruthModule);
}
}








PYBIND11_PLUGIN(_ground_truth) {
    py::module groundTruthModule("_ground_truth", "ground truth submodule of nifty python bindings");

    using namespace nifty::ground_truth;

        
    exportPostProcessCarvingNeuroGroundTruth(groundTruthModule);
    exportOverlap(groundTruthModule);
    exportPartitionComparison(groundTruthModule);
        
    return groundTruthModule.ptr();
}
