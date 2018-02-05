#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{


    void exportGridRag(py::module &);
    void exportGraphAccumulator(py::module &);
    void exportProjectToPixels(py::module &);
    void exportAccumulate(py::module &);

}
}


PYBIND11_MODULE(_rag, ragModule) {

    py::options options;
    options.disable_function_signatures();
        
    ragModule.doc() = "rag submodule of nifty.graph";

    using namespace nifty::graph;

    exportGridRag(ragModule);
    exportGraphAccumulator(ragModule);
    exportProjectToPixels(ragModule);
    exportAccumulate(ragModule);

}

