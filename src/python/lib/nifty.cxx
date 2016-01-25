#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;


namespace nifty{
namespace graph{
    void initSubmoduleGraph(py::module & );
}
}

PYBIND11_PLUGIN(nifty) {
    py::module niftyModule("nifty", "nifty python bindings");

    using namespace nifty;
    graph::initSubmoduleGraph(niftyModule);
}
