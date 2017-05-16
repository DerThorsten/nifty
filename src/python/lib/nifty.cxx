#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <sstream>

#include "nifty/python/converter.hxx"

namespace py = pybind11;

#ifdef WITH_GUROBI
    #include <gurobi_c++.h>
#endif


PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);


struct Configuration{

};


PYBIND11_PLUGIN(_nifty) {
    py::module niftyModule("_nifty", "nifty python bindings");

    using namespace nifty;

    #ifdef WITH_GUROBI
        // Translate Gurobi exceptions to Python exceptions
        // (Must do this explicitly since GRBException doesn't inherit from std::exception)
        static py::exception<GRBException> exc(niftyModule, "GRBException");
        py::register_exception_translator([](std::exception_ptr p) {
            try {
                if (p) std::rethrow_exception(p);
            } catch (const GRBException &e) {
                std::ostringstream ss;
                ss << e.getMessage() << " (Error code:" << e.getErrorCode() << ")";
                exc(ss.str().c_str());
            }
        });
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
        .def_property_readonly_static("WITH_LP_MP", [](py::object /* self */) { 
            #ifdef  WITH_LP_MP
            return true;
            #else
            return false;
            #endif
        })
        .def_property_readonly_static("WITH_QPBO", [](py::object /* self */) { 
            #ifdef  WITH_QPBO
            return true;
            #else
            return false;
            #endif
        })
        
        ;
    return niftyModule.ptr();
}
