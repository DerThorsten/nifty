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
namespace ho_multicut{

    void exportHoMulticutObjective(py::module &);
    void exportHoMulticutFactory(py::module &);
    void exportHoMulticutVisitorBase(py::module &);
    void exportHoMulticutBase(py::module &);


    void exportHoMulticutIlp(py::module &);
    void exportHoMulticutDualDecomposition(py::module &);

    void exportFusionMove(py::module &);
}
}
}
}

PYBIND11_MODULE(_ho_multicut, hoMulticutModule) {

    xt::import_numpy();


    py::options options;
    options.disable_function_signatures();
    
    hoMulticutModule.doc() = "multicut submodule of nifty.graph";
    
    using namespace nifty::graph::opt::ho_multicut;

    exportHoMulticutObjective(hoMulticutModule);
    exportHoMulticutFactory(hoMulticutModule);
    exportHoMulticutVisitorBase(hoMulticutModule);
    exportHoMulticutBase(hoMulticutModule);
    exportHoMulticutIlp(hoMulticutModule);
    exportHoMulticutDualDecomposition(hoMulticutModule);
    exportFusionMove(hoMulticutModule);
    #ifdef WITH_LP_MP
    //exportMulticutMp(multicutModule);
    #endif

}

