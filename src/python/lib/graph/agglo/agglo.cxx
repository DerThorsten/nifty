#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
namespace agglo{

    void exportAgglomerativeClustering(py::module &);    
}
}
}



PYBIND11_PLUGIN(_agglo) {
    py::module aggloModule("_agglo", "agglo submodule of nifty.graph");
    
    using namespace nifty::graph::agglo;

    exportAgglomerativeClustering(aggloModule);

    return aggloModule.ptr();
}

