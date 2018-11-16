#include <pybind11/pybind11.h>
#include <iostream>

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pyvectorize.hpp"

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




PYBIND11_MODULE(_minstcut, minstcutModule) {

    xt::import_numpy();
    py::options options;
    options.disable_function_signatures();

    minstcutModule.doc() = "minstcut submodule of nifty.graph";

    using namespace nifty::graph::opt::minstcut;

    exportMinstcutObjective(minstcutModule);
    exportMinstcutVisitorBase(minstcutModule);
    exportMinstcutBase(minstcutModule);
    exportMinstcutFactory(minstcutModule);
}
