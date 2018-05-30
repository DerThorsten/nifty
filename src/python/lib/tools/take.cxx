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
                for(size_t i = 0; i < shape[0]; ++i){
                    out(i) = relabeling(toRelabel(i));
                }
            }
            return out;
        }, py::arg("relabeling"), py::arg("toRelabel"));



        toolsModule.def("_takeDict",
        [](const std::map<T, T> & relabeling,
           const xt::pytensor<T, 1> & toRelabel
        ){
            typedef typename xt::pytensor<T, 1>::shape_type ShapeType;
            ShapeType shape;
            shape[0] = toRelabel.shape()[0];
            xt::pytensor<T, 1> out = xt::zeros<T>(shape);
            {
                py::gil_scoped_release allowThreads;
                for(size_t i = 0; i < shape[0]; ++i){
                    out(i) = relabeling.at(toRelabel(i));
                }
            }
            return out;
        }, py::arg("relabeling"), py::arg("toRelabel"));

        toolsModule.def("_unique",
        [](const xt::pytensor<T, 1> & values) {
            std::unordered_set<T> uniques;
            {
                py::gil_scoped_release allowThreads;
                for(size_t ii = 0; ii < values.shape()[0]; ++ii) {
                    uniques.insert(values(ii));
                }
            }
            typedef typename xt::pytensor<T, 1>::shape_type Shape;
            Shape shape = {uniques.size()};
            xt::pytensor<T, 1> out = xt::zeros<T>(shape);
            {
                py::gil_scoped_release allowThreads;
                size_t ii = 0;
                for(const T val: uniques) {
                    out(ii) = val;
                    ++ii;
                }
            }
            return out;
        }, py::arg("values"));
    }


    void exportTake(py::module & toolsModule) {
        exportTakeT<uint32_t>(toolsModule);
        exportTakeT<uint64_t>(toolsModule);
        exportTakeT<int32_t>(toolsModule);
        exportTakeT<int64_t>(toolsModule);
    }

}
}
