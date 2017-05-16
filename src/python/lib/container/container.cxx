#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace nifty{
namespace container{
    void exportVector(py::module &);
}
}

    
PYBIND11_PLUGIN(_container) {
    py::module containerModule("_container","container submodule");
    
    using namespace nifty::container;

    exportVector(containerModule);

    return containerModule.ptr();
}
