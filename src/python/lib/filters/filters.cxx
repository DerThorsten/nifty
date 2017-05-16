#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace nifty{
namespace filters{
    void exportAffinities(py::module &);
    void exportGaussianCurvature(py::module &);
}
}

    
PYBIND11_PLUGIN(_filters) {
    py::module filtersModule("_filters","filters submodule");
    
    using namespace nifty::filters;

    exportAffinities(filtersModule);
    exportGaussianCurvature(filtersModule);

    return filtersModule.ptr();
}
