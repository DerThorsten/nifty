#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);


namespace nifty{
namespace graph{


    void exportUndirectedListGraph(py::module &);
    void exportEdgeContractionGraphUndirectedGraph(py::module & );

    void initSubmoduleMulticut(py::module &);
    namespace lifted_multicut{
        void initSubmoduleLiftedMulticut(py::module &);
    }


    void initSubmoduleGala(py::module &);

}
}


PYBIND11_PLUGIN(_graph) {
    py::module graphModule("_graph", "graph submodule of nifty");

    using namespace nifty::graph;

        

    exportUndirectedListGraph(graphModule);
    exportEdgeContractionGraphUndirectedGraph(graphModule);

    initSubmoduleMulticut(graphModule);
    lifted_multicut::initSubmoduleLiftedMulticut(graphModule);
    initSubmoduleGala(graphModule);

        
    return graphModule.ptr();
}

