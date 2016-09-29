#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{


    void exportGalaMainClass(py::module &);
    void exportGalaFeatureBase(py::module &);
    
}
}



PYBIND11_PLUGIN(_gala) {
    py::module galaModule("_gala", "gala submodule of nifty.graph");
    
    using namespace nifty::graph;

    exportGalaMainClass(galaModule);
    exportGalaFeatureBase(galaModule);

    return galaModule.ptr();
}

