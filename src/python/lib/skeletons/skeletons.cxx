#ifdef WITH_Z5
#include <pybind11/pybind11.h>

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"

#include <iostream>


namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);


namespace nifty{
namespace skeletons{

    void exportEvaluation(py::module &);

}
}


PYBIND11_MODULE(_skeletons, module) {

    xt::import_numpy();

    py::options options;
    options.disable_function_signatures();

    module.doc() = "skeletons submodule of nifty";

    using namespace nifty::skeletons;
    exportEvaluation(module);
}
#endif
