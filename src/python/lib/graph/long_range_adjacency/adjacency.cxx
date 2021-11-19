#include <pybind11/pybind11.h>
#include <iostream>

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{

    void exportLongRangeAdjacency(py::module &);
    void exportLongRangeFeatures(py::module &);
    void exportAccumulateLongRangeAffinities(py::module &);
}
}


PYBIND11_PLUGIN(_long_range_adjacency) {

    xt::import_numpy();

    py::options options;
    options.disable_function_signatures();

    py::module module("_long_range_adjacency", "long range adjacency submodule of nifty.graph");

    using namespace nifty::graph;

    exportLongRangeAdjacency(module);
    exportLongRangeFeatures(module);
    exportAccumulateLongRangeAffinities(module);

    return module.ptr();
}

