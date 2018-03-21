#include <pybind11/pybind11.h>
#include <iostream>

// IMPORTANT: This define needs to happen the first time that pyarray is
// imported, i.e. RIGHT HERE !
#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"

namespace py = pybind11;


namespace nifty {
namespace nz5 {
    void exportDatasetWrappers(py::module &);
    void exportUpsampling(py::module &);
}
}


PYBIND11_MODULE(_z5, module) {

    xt::import_numpy();
    module.doc() = "nifty z5 pythonbindings";

    using namespace nifty::nz5;
    exportDatasetWrappers(module);
    exportUpsampling(module);
}
