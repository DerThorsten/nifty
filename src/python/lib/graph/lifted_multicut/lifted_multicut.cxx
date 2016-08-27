#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
namespace lifted_multicut{

    void exportLiftedMulticutObjective(py::module &);



    void initSubmoduleLiftedMulticut(py::module &graphModule) {

        auto liftedMulticutModule = graphModule.def_submodule("lifted_multicut","lifted multicut submodule");
        exportLiftedMulticutObjective(liftedMulticutModule);
    }

}
}
}
