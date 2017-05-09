#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{



}
}



PYBIND11_PLUGIN(_optimization) {
    py::module optimizationModule("_optimization", "optimization submodule of nifty.graph");
    
    using namespace nifty::graph;
    return optimizationModule.ptr();
}

