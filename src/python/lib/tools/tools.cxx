#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);


namespace nifty{
namespace tools{

    void exportMakeDense(py::module &);
    void exportBlocking(py::module &);
    void exportTake(py::module &);
    void exportChangeablePriorityQueue(py::module &);
}
}


PYBIND11_MODULE(_tools, toolsModule) {

    py::options options;
    options.disable_function_signatures();
    
    toolsModule.doc() = "tools submodule of nifty";

    using namespace nifty::tools;

    exportMakeDense(toolsModule);
    exportBlocking(toolsModule);
    exportTake(toolsModule);
    exportChangeablePriorityQueue(toolsModule);
    
}
