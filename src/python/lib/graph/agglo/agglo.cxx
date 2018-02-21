#include <pybind11/pybind11.h>
#include <iostream>

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pyvectorize.hpp"


namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
namespace agglo{
    
    void exportMergeRules(py::module &);    
    void exportAgglomerativeClustering(py::module &);    
    void exportFixationAgglomerativeClustering(py::module &);    
    void exportDualAgglomerativeClustering(py::module &);    
    void exportBsdAgglomerativeClustering(py::module &);    
    void exportLiftedAgglomerativeClusteringPolicy(py::module &);    
}
}
}



PYBIND11_MODULE(_agglo, module) {

    xt::import_numpy();
    
    py::options options;
    options.disable_function_signatures();
    
    module.doc() = "agglo submodule of nifty.graph";

    
    using namespace nifty::graph::agglo;
    exportMergeRules(module);
    exportAgglomerativeClustering(module);
    exportFixationAgglomerativeClustering(module);
    exportDualAgglomerativeClustering(module);
    exportBsdAgglomerativeClustering(module);
    exportLiftedAgglomerativeClusteringPolicy(module);

}

