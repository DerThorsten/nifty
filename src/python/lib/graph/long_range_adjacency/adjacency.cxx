#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{

    void exportLongRangeAdjacency(py::module &);
    void exportLongRangeFeatures(py::module &);
}
}


PYBIND11_PLUGIN(_long_range_adjacency) {

    py::options options;
    options.disable_function_signatures();

    py::module module("_long_range_adjacency", "long range adjacency submodule of nifty.graph");

    using namespace nifty::graph;

    exportLongRangeAdjacency(module);
    exportLongRangeFeatures(module);

    return module.ptr();
}

