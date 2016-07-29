#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
namespace py = pybind11;



namespace nifty{
namespace graph{
    void initSubmoduleGraph(py::module & );
}
}

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);


struct Configuration{

};


PYBIND11_PLUGIN(_nifty) {
    py::module niftyModule("_nifty", "nifty python bindings");

    using namespace nifty;
    graph::initSubmoduleGraph(niftyModule);


py::class_<Configuration>(niftyModule, "Configuration")
    .def_property_readonly_static("WITH_CPLEX", [](py::object /* self */) { 
        #ifdef  WITH_CPLEX
        return true;
        #else
        return false;
        #endif
    })
    .def_property_readonly_static("WITH_GUROBI", [](py::object /* self */) { 
        #ifdef  WITH_GUROBI
        return true;
        #else
        return false;
        #endif
    })
    .def_property_readonly_static("WITH_GLPK", [](py::object /* self */) { 
        #ifdef  WITH_GLPK
        return true;
        #else
        return false;
        #endif
    })
    .def_property_readonly_static("WITH_HDF5", [](py::object /* self */) { 
        #ifdef  WITH_HDF5
        return true;
        #else
        return false;
        #endif
    })
    ;
}
