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
    
    return minstcutModule.ptr();
}

