#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <typeinfo> // to debug atm

#include "xtensor-python/pyarray.hpp"
#include "nifty/tools/make_dense.hxx"

namespace py = pybind11;



namespace nifty{
namespace tools{

    template<class T>
    void exportMakeDenseT(py::module & toolsModule) {

        toolsModule.def("makeDense",
        [](const xt::pyarray<T> & dataIn){

            typedef typename xt::pyarray<T>::shape_type ShapeType;
            ShapeType shape(dataIn.shape().begin(), dataIn.shape().end());
            xt::pyarray<T> dataOut(shape);
            {
                py::gil_scoped_release allowThreads;
                tools::makeDense(dataIn, dataOut);
            }
            return dataOut;
        });
    }


    void exportMakeDense(py::module & toolsModule) {

        exportMakeDenseT<uint32_t>(toolsModule);
        exportMakeDenseT<uint64_t>(toolsModule);
        exportMakeDenseT<int32_t>(toolsModule);

        //exportMakeDenseT<float   , false>(toolsModule);
        exportMakeDenseT<int64_t>(toolsModule);
    }

}
}
