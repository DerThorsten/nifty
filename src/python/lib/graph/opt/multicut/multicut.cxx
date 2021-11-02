#include <pybind11/pybind11.h>
#include <iostream>

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);



namespace nifty{
namespace graph{
namespace opt{
namespace multicut{

    void exportMulticutObjective(py::module &);
    void exportMulticutFactory(py::module &);
    void exportMulticutVisitorBase(py::module &);
    void exportMulticutBase(py::module &);

    void exportMulticutIlp(py::module &);
    void exportCgc(py::module &);
    void exportMulticutGreedyAdditive(py::module &);
    void exportMulticutGreedyFixation(py::module &);
    void exportFusionMoveBased(py::module &);
    void exportPerturbAndMap(py::module &);
    void exportMulticutDecomposer(py::module &);
    void exportChainedSolvers(py::module &);
    void exportMulticutCcFusionMoveBased(py::module &);
    void exportKernighanLin(py::module &);
    #if WITH_LP_MP
    void exportMulticutMp(py::module &);
    #endif
}
}
}
}

PYBIND11_MODULE(_multicut, multicutModule) {

    xt::import_numpy();

    py::options options;
    options.disable_function_signatures();
    multicutModule.doc() = "multicut submodule of nifty.graph";
    using namespace nifty::graph::opt::multicut;

    exportMulticutObjective(multicutModule);
    exportMulticutVisitorBase(multicutModule);
    exportMulticutBase(multicutModule);
    exportMulticutFactory(multicutModule);
    exportMulticutIlp(multicutModule);
    exportCgc(multicutModule);
    exportMulticutGreedyAdditive(multicutModule);
    exportMulticutGreedyFixation(multicutModule);
    exportFusionMoveBased(multicutModule);
    exportPerturbAndMap(multicutModule);
    exportMulticutDecomposer(multicutModule);
    exportChainedSolvers(multicutModule);
    exportMulticutCcFusionMoveBased(multicutModule);
    exportKernighanLin(multicutModule);

    #ifdef WITH_LP_MP
    exportMulticutMp(multicutModule);
    #endif

}
