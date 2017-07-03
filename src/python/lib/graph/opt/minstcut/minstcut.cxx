#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
namespace opt{
namespace minstcut{

    void exportMinstcutObjective(py::module &);
    void exportMinstcutFactory(py::module &);
    void exportMinstcutVisitorBase(py::module &);
    void exportMinstcutBase(py::module &);
    #if WITH_QPBO
    void exportMinstcutQpbo(py::module &);
    #endif 

    void exportMinstcutCcFusionMoveBased(py::module &);
    #if WITH_QPBO
    void exportMinstcutGreedyAdditive(py::module &);
    #endif 

} // namespace nifty::graph::opt::minstcut
} // namespace nifty::graph::opt
}
}




PYBIND11_PLUGIN(_minstcut) {

    py::options options;
    options.disable_function_signatures();
    
    py::module minstcutModule("_minstcut", "minstcut submodule of nifty.graph");
    
    using namespace nifty::graph::opt::minstcut;

    exportMinstcutObjective(minstcutModule);
    exportMinstcutVisitorBase(minstcutModule);
    exportMinstcutBase(minstcutModule);
    exportMinstcutFactory(minstcutModule);
    #ifdef WITH_QPBO
    exportMinstcutQpbo(minstcutModule);
    exportMinstcutGreedyAdditive(minstcutModule);
    #endif
    exportMinstcutCcFusionMoveBased(minstcutModule);
    return minstcutModule.ptr();
}

