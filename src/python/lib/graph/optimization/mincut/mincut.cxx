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
    void exportMincutQpbo(py::module &);
    namespace mincut{
        void exportMincutCcFusionMoveBased(py::module &);
        void exportMincutGreedyAdditive(py::module &);
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
    #endif
    mincut::exportMincutCcFusionMoveBased(mincutModule);
    mincut::exportMincutGreedyAdditive(mincutModule);

    return mincutModule.ptr();
}

