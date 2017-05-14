#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);



namespace nifty{
namespace graph{



    void exportMulticutObjective(py::module &);
    void exportMulticutFactory(py::module &);
    void exportMulticutVisitorBase(py::module &);
    void exportMulticutBase(py::module &);

    namespace optimization{
    namespace multicut{
        void exportMulticutIlp(py::module &);
    }
    }
    void exportMulticutGreedyAdditive(py::module &);
    void exportFusionMoveBased(py::module &);
    void exportPerturbAndMap(py::module &);
    void exportMulticutDecomposer(py::module &);
    //void exportMulticutAndres(py::module &);
    void exportChainedSolvers(py::module &);
    
    #if WITH_QPBO
    void exportCgc(py::module &);
    #endif
    void exportBlockMulticut(py::module &);
    
    #if WITH_LP_MP
    void exportMulticutMp(py::module &);
    #endif
}
}



PYBIND11_PLUGIN(_multicut) {
    py::module multicutModule("_multicut", "multicut submodule of nifty.graph");
    
    using namespace nifty::graph;

    exportMulticutObjective(multicutModule);
    exportMulticutVisitorBase(multicutModule);
    exportMulticutBase(multicutModule);
    exportMulticutFactory(multicutModule);
    nifty::graph::optimization::multicut::exportMulticutIlp(multicutModule);
    exportMulticutGreedyAdditive(multicutModule);
    exportFusionMoveBased(multicutModule);
    exportPerturbAndMap(multicutModule);
    exportMulticutDecomposer(multicutModule);
    //exportMulticutAndres(multicutModule);
    exportChainedSolvers(multicutModule);
    exportBlockMulticut(multicutModule);
    
    #ifdef WITH_LP_MP
    exportMulticutMp(multicutModule);
    #endif
    
    #ifdef WITH_QPBO
    exportCgc(multicutModule);
    #endif
    
    return multicutModule.ptr();
}

