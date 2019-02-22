#include <pybind11/pybind11.h>

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"

namespace py = pybind11;

namespace nifty{
namespace filters{
    void exportGaussianCurvature(py::module &);
    void exportNonMaximumSuppression(py::module &);
}
}


PYBIND11_MODULE(_filters, filtersModule) {
    xt::import_numpy();
    py::options options;
    options.disable_function_signatures();
    filtersModule.doc() = "filters submodule";
    using namespace nifty::filters;
    exportGaussianCurvature(filtersModule);
    exportNonMaximumSuppression(filtersModule);
}
