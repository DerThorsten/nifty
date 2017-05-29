#include <pybind11/pybind11.h>
#include <iostream>



namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);


namespace nifty{
namespace cgp{

    void exportTopologicalGrid(py::module &);
    void exportBounds(py::module &);
    void exportGeometry(py::module &);
    // FIXME this does not build right now because it
    // pulls in boost::math which is not c++14 ready (at least not in boost 1.61)
    void exportFeatures(py::module &);
}
}


PYBIND11_PLUGIN(_cgp) {
    py::module cgpModule("_cgp", "cgp submodule of nifty");

    using namespace nifty::cgp;

    exportTopologicalGrid(cgpModule);
    exportBounds(cgpModule);
    exportGeometry(cgpModule);
    exportFeatures(cgpModule);

    return cgpModule.ptr();
}
