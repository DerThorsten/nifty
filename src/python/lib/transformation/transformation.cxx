#include <pybind11/pybind11.h>

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#include "xtensor-python/pyvectorize.hpp"

#include <iostream>

namespace py = pybind11;


namespace nifty{
namespace transformation{
    void exportAffineTransformation(py::module &);
    // void exportCoordinateTransformation(py::module &);
    #ifdef WITH_Z5
    void exportAffineTransformationZ5(py::module &);
    void exportCoordinateTransformationZ5(py::module &);
    #endif
    #ifdef WITH_HDF5
    void exportAffineTransformationH5(py::module &);
    // void exportCoordinateTransformationH5(py::module &);
    #endif
}
}


PYBIND11_MODULE(_transformation, module) {

    xt::import_numpy();

    py::options options;
    options.disable_function_signatures();

    module.doc() = "transformation submodule of nifty";

    using namespace nifty::transformation;
    exportAffineTransformation(module);
    // exportCoordinateTransformation(module);
    #ifdef WITH_Z5
    exportAffineTransformationZ5(module);
    exportCoordinateTransformationZ5(module);
    #endif
    #ifdef WITH_HDF5
    exportAffineTransformationH5(module);
    // exportCoordinateTransformationH5(module);
    #endif
}
