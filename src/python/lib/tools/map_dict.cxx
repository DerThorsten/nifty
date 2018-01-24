#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "xtensor-python/pytensor.hpp"
#include "nifty/tools/array_tools.hxx"

namespace py = pybind11;



namespace nifty{
namespace tools{

    template<class T, unsigned DIM>
    void exportMapDictionaryToArrayT(py::module & toolsModule) {

        toolsModule.def("mapDictionaryToArray",
        [](xt::pytensor<T, DIM> & data, std::map<T, T> dict, bool haveIgnoreValue, T ignoreValue){

            py::gil_scoped_release allowThreads;
            mapDictionaryToArray<DIM>(data, dict, haveIgnoreValue, ignoreValue);

        }, py::arg("data"), py::arg("dict"), py::arg("haveIgnoreValue")=false, py::arg("ignoreValue")=0);


        toolsModule.def("mapLabelingToArray",
        [](xt::pytensor<T, DIM> & data, xt::pytensor<T, 1> labeling, bool haveIgnoreValue, T ignoreValue){

            py::gil_scoped_release allowThreads;
            mapLabelingToArray<DIM>(data, labeling, haveIgnoreValue, ignoreValue);

        }, py::arg("data"), py::arg("labeling"), py::arg("haveIgnoreValue")=false, py::arg("ignoreValue")=0);
    }


    void exportMapDictionaryToArray(py::module & toolsModule) {

        exportMapDictionaryToArrayT<uint32_t, 2>(toolsModule);
        exportMapDictionaryToArrayT<uint32_t, 3>(toolsModule);

        exportMapDictionaryToArrayT<uint64_t, 2>(toolsModule);
        exportMapDictionaryToArrayT<uint64_t, 3>(toolsModule);

        exportMapDictionaryToArrayT<int32_t, 2>(toolsModule);
        exportMapDictionaryToArrayT<int32_t, 3>(toolsModule);

        exportMapDictionaryToArrayT<int64_t, 2>(toolsModule);
        exportMapDictionaryToArrayT<int64_t, 3>(toolsModule);
    }

}
}
