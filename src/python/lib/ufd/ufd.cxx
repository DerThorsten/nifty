#include <pybind11/pybind11.h>

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"

namespace py = pybind11;

namespace nifty{
namespace ufd{
    void exportUfd(py::module &);
    void exportBoostUfd(py::module &);
}
}


PYBIND11_MODULE(_ufd, ufdModule) {

    xt::import_numpy();
    py::options options;
    options.disable_function_signatures();
    ufdModule.doc() = "ufd submodule";

    using namespace nifty::ufd;

    exportUfd(ufdModule);
    exportBoostUfd(ufdModule);
}
