#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);



namespace nifty{
namespace graph{
namespace optimization{
namespace multicut{

    void exportMulticutObjective(py::module &);
    void exportMulticutFactory(py::module &);
    void exportMulticutVisitorBase(py::module &);
    void exportMulticutBase(py::module &);

    // we are currently  refactoring in little 
    // pieces: optimization::multicut is the new cool namespace
    // implemented by Nish

    void exportMulticutIlp(py::module &);
    void exportCgc(py::module &);
    void exportMulticutGreedyAdditive(py::module &);
    void exportFusionMoveBased(py::module &);
    void exportPerturbAndMap(py::module &);
    void exportMulticutDecomposer(py::module &);
    void exportMulticutAndres(py::module &);
    void exportChainedSolvers(py::module &);
    void exportMulticutCcFusionMoveBased(py::module &);
    
    #if WITH_LP_MP
    void exportMulticutMp(py::module &);
    #endif
}
}
}
}

PYBIND11_PLUGIN(_multicut) {

    py::options options;
    options.disable_function_signatures();
    
    py::module multicutModule("_multicut", "multicut submodule of nifty.graph");
    
    using namespace nifty::graph::optimization::multicut;

    exportMulticutObjective(multicutModule);
    exportMulticutVisitorBase(multicutModule);
    exportMulticutBase(multicutModule);
    exportMulticutFactory(multicutModule);
    exportMulticutIlp(multicutModule);
    exportCgc(multicutModule);
    exportMulticutGreedyAdditive(multicutModule);
    exportFusionMoveBased(multicutModule);
    exportPerturbAndMap(multicutModule);
    exportMulticutDecomposer(multicutModule);
    exportMulticutAndres(multicutModule);
    exportChainedSolvers(multicutModule);
    exportMulticutCcFusionMoveBased(multicutModule);

    
    #ifdef WITH_LP_MP
    exportMulticutMp(multicutModule);
    #endif
    
    
    return multicutModule.ptr();
}

