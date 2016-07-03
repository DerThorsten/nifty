#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{


    void exportGalaMainClass(py::module &);
    void exportGalaFeatureBase(py::module &);
    
    void initSubmoduleGala(py::module &graphModule) {

        auto galaModule = graphModule.def_submodule("gala","gala submodule");
        exportGalaMainClass(galaModule);
        exportGalaFeatureBase(galaModule);
    }

}
}
