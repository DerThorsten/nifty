#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>

#include "nifty/z5/z5.hxx"

namespace py = pybind11;

namespace nifty {
namespace nz5 {

    template<class T>
    void exportDatasetWrapperT(py::module & module, const std::string & dsName) {
        typedef DatasetWrapper<T> clsT;
        py::class_<clsT>(module, dsName.c_str())
            .def(py::init([](const std::string & pathToFile, const std::string & key){
                return new clsT(pathToFile, key);
            }))
            .def_property_readonly("shape", &clsT::shape);
        ;
    }

    void exportDatasetWrappers(py::module & module) {
        // uint types
        exportDatasetWrapperT<uint8_t>(module,  "DatasetWrapperUint8");
        exportDatasetWrapperT<uint16_t>(module, "DatasetWrapperUint16");
        exportDatasetWrapperT<uint32_t>(module, "DatasetWrapperUint32");
        exportDatasetWrapperT<uint64_t>(module, "DatasetWrapperUint64");
        // int types
        exportDatasetWrapperT<int8_t>(module,  "DatasetWrapperInt8");
        exportDatasetWrapperT<int16_t>(module, "DatasetWrapperInt16");
        exportDatasetWrapperT<int32_t>(module, "DatasetWrapperInt32");
        exportDatasetWrapperT<int64_t>(module, "DatasetWrapperInt64");
        // float types
        exportDatasetWrapperT<float>(module, "DatasetWrapperFloat32");
        exportDatasetWrapperT<double>(module, "DatasetWrapperFloat64");
    }
}
}
