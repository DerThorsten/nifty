#include <pybind11/pybind11.h>
#include <iostream>

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pytensor.hpp"
#include "xtensor-python/pyarray.hpp"

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
    void exportChangeablePriorityQueue(py::module &);
    void exportMapDictionaryToArray(py::module &);
}
}


PYBIND11_MODULE(_tools, toolsModule) {

    xt::import_numpy();

    py::options options;
    options.disable_function_signatures();
    toolsModule.doc() = "tools submodule of nifty";

    using namespace nifty::tools;

    exportMakeDense(toolsModule);
    exportBlocking(toolsModule);
    exportTake(toolsModule);
    exportUnique(toolsModule);
    exportNodesToBlocks(toolsModule);
    exportEdgeMapping(toolsModule);
    exportSleep(toolsModule);
    exportChangeablePriorityQueue(toolsModule);
    exportMapDictionaryToArray(toolsModule);
}
