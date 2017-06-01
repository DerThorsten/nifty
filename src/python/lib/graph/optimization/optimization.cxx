#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
namespace optimization{

} // namespace nifty::graph::optimization
}
}



PYBIND11_PLUGIN(_optimization) {
    py::module optimizationModule("_optimization", "optimization submodule of nifty.graph");
    
    using namespace nifty::graph::optimization;
    return optimizationModule.ptr();
}

