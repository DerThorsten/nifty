#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);


namespace nifty{
namespace tools{

    void exportMakeDense(py::module &);


}
}


PYBIND11_PLUGIN(_tools) {
    py::module toolsModule("_tools", "tools submodule of nifty");

    using namespace nifty::tools;

    exportMakeDense(toolsModule);

        
    return toolsModule.ptr();
}
