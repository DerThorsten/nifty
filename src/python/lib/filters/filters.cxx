#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace nifty{
namespace filters{
    void exportAffinities(py::module &);
    void exportGaussianCurvature(py::module &);
}
}

    
PYBIND11_MODULE(_filters, filtersModule) {
    py::options options;
    options.disable_function_signatures();
    
    filtersModule.doc() = "filters submodule";
    
    using namespace nifty::filters;

    exportAffinities(filtersModule);
    exportGaussianCurvature(filtersModule);

}
