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
    void exportMulticutIlp(py::module &);
    void exportMulticutGreedyAdditive(py::module &);
    void exportFusionMoveBased(py::module &);
    void exportMulticutMp(py::module &);
    void exportMulticutKernighanLin(py::module &);
    void exportPerturbAndMap(py::module &);
    void exportMulticutDecomposer(py::module &);
    void exportCgc(py::module &);
    void exportMulticutAndres(py::module &);


}
}



PYBIND11_PLUGIN(_multicut) {
    py::module multicutModule("_multicut", "multicut submodule of nifty.graph");
    
    using namespace nifty::graph;

    exportMulticutObjective(multicutModule);
    exportMulticutVisitorBase(multicutModule);
    exportMulticutBase(multicutModule);
    exportMulticutFactory(multicutModule);
    exportMulticutIlp(multicutModule);
    exportMulticutGreedyAdditive(multicutModule);
    exportFusionMoveBased(multicutModule);
    exportPerturbAndMap(multicutModule);
    exportMulticutKernighanLin(multicutModule);
    exportMulticutDecomposer(multicutModule);
    #ifdef WITH_LP_MP
    exportMulticutMp(multicutModule);
    #endif
    #ifdef WITH_QPBO
    exportCgc(multicutModule);
    #endif
    exportMulticutAndres(multicutModule);

    return multicutModule.ptr();
}

