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
    void exportPerturbAndMap(py::module &);


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

    return multicutModule.ptr();
}

