#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
namespace opt{
namespace lifted_multicut{

    void exportLiftedMulticutObjective(py::module &);
    void exportLiftedMulticutFactory(py::module &);
    void exportLiftedMulticutVisitorBase(py::module &);
    void exportLiftedMulticutBase(py::module &);
    void exportLiftedMulticutGreedyAdditive(py::module &);
    void exportLiftedMulticutKernighanLin(py::module &);
    void exportLiftedMulticutIlp(py::module &);
    void exportLiftedMulticutMp(py::module &);
    //void exportLiftedMulticutAndresKernighanLin(py::module &);
    void exportLiftedMulticutAndresGreedyAdditive(py::module &);
    void exportFusionMoveBased(py::module &);
    void exportLiftedGraphFeatures(py::module &);

}
} // namespace nifty::graph::opt
}
}



PYBIND11_PLUGIN(_lifted_multicut) {


    py::options options;
    options.disable_function_signatures();
    
    py::module liftedMulticutModule("_lifted_multicut", "lifted_multicut submodule of nifty.graph");
    
    using namespace nifty::graph::opt::lifted_multicut;

    exportLiftedMulticutObjective(liftedMulticutModule);
    exportLiftedMulticutVisitorBase(liftedMulticutModule);
    exportLiftedMulticutBase(liftedMulticutModule);
    exportLiftedMulticutFactory(liftedMulticutModule);
    exportLiftedMulticutGreedyAdditive(liftedMulticutModule);
    exportLiftedMulticutKernighanLin(liftedMulticutModule);
    exportLiftedMulticutIlp(liftedMulticutModule);
    //exportLiftedMulticutAndresKernighanLin(liftedMulticutModule);
    exportLiftedMulticutAndresGreedyAdditive(liftedMulticutModule);
    exportFusionMoveBased(liftedMulticutModule);
    exportLiftedGraphFeatures(liftedMulticutModule);
    
    #ifdef WITH_LP_MP
    exportLiftedMulticutMp(liftedMulticutModule);
    #endif

    return liftedMulticutModule.ptr();
}
