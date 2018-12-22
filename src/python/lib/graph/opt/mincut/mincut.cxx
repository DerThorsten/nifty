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
namespace mincut{

    void exportMincutObjective(py::module &);
    void exportMincutFactory(py::module &);
    void exportMincutVisitorBase(py::module &);
    void exportMincutBase(py::module &);
    void exportMincutCcFusionMoveBased(py::module &);
    #if WITH_QPBO
    void exportMincutQpbo(py::module &);
    void exportMincutGreedyAdditive(py::module &);
    #endif

} // namespace nifty::graph::opt::mincut
} // namespace nifty::graph::opt
}
}


PYBIND11_MODULE(_mincut, mincutModule) {

    xt::import_numpy();

    py::options options;
    options.disable_function_signatures();

    mincutModule.doc() = "mincut submodule of nifty.graph";

    using namespace nifty::graph::opt::mincut;

    exportMincutObjective(mincutModule);
    exportMincutVisitorBase(mincutModule);
    exportMincutBase(mincutModule);
    exportMincutFactory(mincutModule);
    exportMincutCcFusionMoveBased(mincutModule);
    #ifdef WITH_QPBO
    exportMincutQpbo(mincutModule);
    exportMincutGreedyAdditive(mincutModule);
    #endif
}
