#include <pybind11/pybind11.h>

#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "nifty/python/converter.hxx"
#include "nifty/tools/array_tools.hxx"

namespace py = pybind11;

namespace nifty{
namespace tools{

    template<class T>
    void exportUniqueListT(py::module & toolsModule) {
        
        toolsModule.def("uniqueList",
        [](
           const std::vector<T> & values
        ){
            std::vector<T> out; 
            {
                py::gil_scoped_release allowThreads;
                uniques(values, out);
            }
            return out;
        });
    }
    
    void exportUnique(py::module & toolsModule) {
        exportUniqueListT<uint32_t>(toolsModule);
        exportUniqueListT<uint64_t>(toolsModule);
    }
}
}
