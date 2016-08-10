#ifdef WITH_HDF5
#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "nifty/python/converter.hxx"
#include "nifty/tools/make_dense.hxx"

namespace py = pybind11;



namespace nifty{
namespace tools{

    template<class T>
    void exportMakeDenseT(py::module & toolsModule) {

        toolsModule.def("makeDense",
        [](
           nifty::marray::PyView<T> dataIn
        ){
            nifty::marray::PyView<T> dataOut(dataIn.shapeBegin(), dataIn.shapeEnd());
            tools::makeDense(dataIn, dataOut);
            return dataOut;
        });
    }


    void exportMakeDense(py::module & toolsModule) {
        exportMakeDenseT<uint32_t>(toolsModule);
    }

}
}

#endif
