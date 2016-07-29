#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
namespace agglo{


    void exportAgglomerativeClustering(py::module &);
    
    void initSubmoduleAgglo(py::module &graphModule) {

        auto aggloModule = graphModule.def_submodule("agglo","agglo submodule");

        exportAgglomerativeClustering(aggloModule);

    }

}
}
}
