#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "xtensor-python/pytensor.hpp"


namespace py = pybind11;


namespace nifty{
namespace tools{

    // TODO copies here are un-necessary, could also do in-place !
    template<class T>
    void exportTakeT(py::module & toolsModule) {

        toolsModule.def("_take",
        [](const xt::pytensor<T, 1> & relabeling,
           const xt::pytensor<T, 1> & toRelabel
        ){
            typedef typename xt::pytensor<T, 1>::shape_type ShapeType;
            ShapeType shape;
            shape[0] = toRelabel.shape()[0];
            xt::pytensor<T, 1> out = xt::zeros<T>(shape);
            {
                py::gil_scoped_release allowThreads;
                for(std::size_t i = 0; i < shape[0]; ++i){
                    out(i) = relabeling(toRelabel(i));
                }
            }
            return out;
        }, py::arg("relabeling"), py::arg("toRelabel"));


        toolsModule.def("_takeDict",
        [](const std::unordered_map<T, T> & relabeling,
           const xt::pytensor<T, 1> & toRelabel
        ){
            typedef typename xt::pytensor<T, 1>::shape_type ShapeType;
            ShapeType shape;
            shape[0] = toRelabel.shape()[0];
            xt::pytensor<T, 1> out = xt::zeros<T>(shape);
            {
                py::gil_scoped_release allowThreads;
                for(std::size_t i = 0; i < shape[0]; ++i){
                    out(i) = relabeling.at(toRelabel(i));
                }
            }
            return out;
        }, py::arg("relabeling"), py::arg("toRelabel"));


        toolsModule.def("inflateLabeling",
        [](const xt::pytensor<T, 1> & values, const xt::pytensor<T, 1> & labels, const T maxVal, const T fillVal){
            const unsigned int nValues = maxVal + 1;
            xt::pytensor<T, 1> out = xt::zeros<T>({nValues});
            {
                py::gil_scoped_release allowThreads;
                T index = 0;
                for(T consecVal = 0; consecVal < nValues; ++consecVal) {
                    if(values[index] == consecVal) {
                        out[consecVal] = labels[index];
                        ++index;
                    } else {
                        out[consecVal] = fillVal;
                    }
                }
            }
            return out;
        }, py::arg("values"), py::arg("labels"), py::arg("maxVal"), py::arg("fillVal")=0);
    }


    void exportTake(py::module & toolsModule) {
        exportTakeT<uint32_t>(toolsModule);
        exportTakeT<uint64_t>(toolsModule);
        exportTakeT<int32_t>(toolsModule);
        exportTakeT<int64_t>(toolsModule);
    }

}
}
