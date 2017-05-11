#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{


    void exportMincutObjective(py::module &);
    void exportMincutFactory(py::module &);
    void exportMincutVisitorBase(py::module &);
    void exportMincutBase(py::module &);
    #if WITH_QPBO
    void exportMincutQpbo(py::module &);
    #endif 
    namespace mincut{
        void exportMincutCcFusionMoveBased(py::module &);
        #if WITH_QPBO
        void exportMincutGreedyAdditive(py::module &);
        #endif 
    }
}
}



PYBIND11_PLUGIN(_mincut) {
    py::module mincutModule("_mincut", "mincut submodule of nifty.graph");
    
    using namespace nifty::graph;

    exportMincutObjective(mincutModule);
    exportMincutVisitorBase(mincutModule);
    exportMincutBase(mincutModule);
    exportMincutFactory(mincutModule);
    #ifdef WITH_QPBO
    exportMincutQpbo(mincutModule);
    mincut::exportMincutGreedyAdditive(mincutModule);
    #endif
    mincut::exportMincutCcFusionMoveBased(mincutModule);
    return mincutModule.ptr();
}

