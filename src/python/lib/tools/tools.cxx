#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);


namespace nifty{
namespace tools{

    void exportMakeDense(py::module &);
    void exportBlocking(py::module &);
    void exportTake(py::module &);
    void exportUnique(py::module &);
    void exportNodesToBlocks(py::module &);
    void exportEdgeMapping(py::module &);
    void exportSleep(py::module &);
}
}


PYBIND11_PLUGIN(_tools) {

    py::options options;
    options.disable_function_signatures();
    
    py::module toolsModule("_tools", "tools submodule of nifty");

    using namespace nifty::tools;

    exportMakeDense(toolsModule);
    exportBlocking(toolsModule);
    exportTake(toolsModule);
    exportUnique(toolsModule);
    exportNodesToBlocks(toolsModule);
    exportEdgeMapping(toolsModule);
    exportSleep(toolsModule);

    return toolsModule.ptr();
}
