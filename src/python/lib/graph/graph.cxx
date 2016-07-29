#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);


namespace nifty{
namespace graph{


    void exportUndirectedGraph(py::module &);
    void exportEdgeContractionGraphUndirectedGraph(py::module & );

    void initSubmoduleMulticut(py::module &);
    void initSubmoduleRag(py::module &);
    namespace agglo{
        void initSubmoduleAgglo(py::module &);
    }
    void initSubmoduleGala(py::module &);

    void initSubmoduleGraph(py::module &niftyModule) {
        auto graphModule = niftyModule.def_submodule("graph","graph submodule");


        exportUndirectedGraph(graphModule);
        exportEdgeContractionGraphUndirectedGraph(graphModule);

        initSubmoduleMulticut(graphModule);
        initSubmoduleRag(graphModule);
        agglo::initSubmoduleAgglo(graphModule);
        initSubmoduleGala(graphModule);
    }

}
}
