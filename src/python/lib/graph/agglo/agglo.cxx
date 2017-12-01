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

    void exportAgglomerativeClustering(py::module &);    
    void exportFixationAgglomerativeClustering(py::module &);    
    void exportLiftedAgglomerativeClusteringPolicy(py::module &);    
    void exportGeneralizedLongRangeClusterPolicy(py::module &);
}
}
}



PYBIND11_PLUGIN(_agglo) {

    xt::import_numpy();
    
    py::options options;
    options.disable_function_signatures();
        
    py::module aggloModule("_agglo", "agglo submodule of nifty.graph");
    
    using namespace nifty::graph::agglo;

    exportAgglomerativeClustering(aggloModule);
    exportFixationAgglomerativeClustering(aggloModule);
    exportLiftedAgglomerativeClusteringPolicy(aggloModule);
    exportGeneralizedLongRangeClusterPolicy(aggloModule);
    return aggloModule.ptr();
}

