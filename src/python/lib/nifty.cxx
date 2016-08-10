#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

#include "nifty/python/converter.hxx"

namespace py = pybind11;



namespace nifty{
namespace graph{
    void initSubmoduleGraph(py::module & );
}
namespace tools{
    void initSubmoduleTools(py::module &);
}
#ifdef WITH_HDF5
namespace hdf5{
    void initSubmoduleHdf5(py::module & );
}
#endif
}

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);


struct Configuration{

};


PYBIND11_PLUGIN(_nifty) {
    py::module niftyModule("_nifty", "nifty python bindings");

    using namespace nifty;

    // get ride of this nonsense
    py::class_<MyNone>(niftyModule, "_MyNone")
        .def(py::init<>())
    ;


    graph::initSubmoduleGraph(niftyModule);
    tools::initSubmoduleTools(niftyModule);

    #ifdef WITH_HDF5
    hdf5::initSubmoduleHdf5(niftyModule);
    #endif

    // \TODO move to another header
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
        .def_property_readonly_static("WITH_HDF52", [](py::object /* self */) { 
            #ifdef  WITH_HDF52
            return true;
            #else
            return false;
            #endif
        })
        ;
}
