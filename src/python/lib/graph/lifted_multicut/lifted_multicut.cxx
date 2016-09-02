#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
namespace lifted_multicut{

    void exportLiftedMulticutObjective(py::module &);
    void exportLiftedMulticutFactory(py::module &);
    void exportLiftedMulticutVisitorBase(py::module &);
    void exportLiftedMulticutBase(py::module &);
    void exportLiftedMulticutGreedyAdditive(py::module &);
    void exportLiftedMulticutIlp(py::module &);

    void initSubmoduleLiftedMulticut(py::module &graphModule) {

        auto liftedMulticutModule = graphModule.def_submodule("lifted_multicut","lifted multicut submodule");
        exportLiftedMulticutObjective(liftedMulticutModule);
        exportLiftedMulticutVisitorBase(liftedMulticutModule);
        exportLiftedMulticutBase(liftedMulticutModule);
        exportLiftedMulticutFactory(liftedMulticutModule);
        exportLiftedMulticutGreedyAdditive(liftedMulticutModule);
        exportLiftedMulticutIlp(liftedMulticutModule);
    }

}
}
}
