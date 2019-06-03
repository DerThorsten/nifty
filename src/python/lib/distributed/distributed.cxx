#ifdef WITH_Z5
#include <pybind11/pybind11.h>

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"

#include <iostream>


namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);


namespace nifty{
namespace distributed{

    void exportGraphExtraction(py::module &);
    void exportDistributedGraph(py::module &);
    void exportMergeableFeatures(py::module &);
    void exportDistributedUtils(py::module &);
    void exportLiftedUtils(py::module &);
    void exportMorphology(py::module &);
    void exportEdgeMorphology(py::module &);
    void exportEvalUtils(py::module &);

}
}


PYBIND11_MODULE(_distributed, module) {

    xt::import_numpy();

    py::options options;
    options.disable_function_signatures();

    module.doc() = "distributed submodule of nifty";

    using namespace nifty::distributed;
    exportGraphExtraction(module);
    exportDistributedGraph(module);
    exportMergeableFeatures(module);
    exportDistributedUtils(module);
    exportLiftedUtils(module);
    exportMorphology(module);
    exportEdgeMorphology(module);
    exportEvalUtils(module);
}
#endif
