#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{


    void exportGridRag(py::module &, py::module &);


    void initSubmoduleRag(py::module &graphModule) {

        auto ragModule = graphModule.def_submodule("rag","rag submodule");
        exportGridRag(ragModule, graphModule);
    }

}
}
