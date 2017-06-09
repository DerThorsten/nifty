#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
namespace optimization{
namespace mincut{

    void exportMincutObjective(py::module &);
    void exportMincutFactory(py::module &);
    void exportMincutVisitorBase(py::module &);
    void exportMincutBase(py::module &);
    #if WITH_QPBO
    void exportMincutQpbo(py::module &);
    #endif 

    void exportMincutCcFusionMoveBased(py::module &);
    #if WITH_QPBO
    void exportMincutGreedyAdditive(py::module &);
    #endif 

} // namespace nifty::graph::optimization::mincut
} // namespace nifty::graph::optimization
}
}




PYBIND11_PLUGIN(_mincut) {

    py::options options;
    options.disable_function_signatures();
    
    py::module mincutModule("_mincut", "mincut submodule of nifty.graph");
    
    using namespace nifty::graph::optimization::mincut;

    exportMincutObjective(mincutModule);
    exportMincutVisitorBase(mincutModule);
    exportMincutBase(mincutModule);
    exportMincutFactory(mincutModule);
    #ifdef WITH_QPBO
    exportMincutQpbo(mincutModule);
    exportMincutGreedyAdditive(mincutModule);
    #endif
    exportMincutCcFusionMoveBased(mincutModule);
    return mincutModule.ptr();
}

