#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);


namespace nifty{
namespace graph{


    void exportUndirectedGraph(py::module &);
    void exportComputeRag(py::module &);

    void initSubmoduleMulticut(py::module &);

    void initSubmoduleGraph(py::module &niftyModule) {
        auto graphModule = niftyModule.def_submodule("graph","graph submodule");

        exportUndirectedGraph(graphModule);
        exportComputeRag(graphModule);
        initSubmoduleMulticut(graphModule);
    }

}
}
