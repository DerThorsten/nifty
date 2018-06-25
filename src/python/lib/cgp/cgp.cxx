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
    void exportCnnRepresentation(py::module &);
}
}


PYBIND11_MODULE(_cgp, cgpModule) {

    py::options options;
    options.disable_function_signatures();

    cgpModule.doc() = "cgp submodule of nifty";

    using namespace nifty::cgp;

    exportTopologicalGrid(cgpModule);
    exportBounds(cgpModule);
    exportGeometry(cgpModule);
    exportFeatures(cgpModule);
    exportCnnRepresentation(cgpModule);

}
