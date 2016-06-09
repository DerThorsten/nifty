#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

#include "converter.hxx"
namespace py = pybind11;



namespace nifty{
namespace graph{
    void initSubmoduleGraph(py::module & );
}
}

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

PYBIND11_PLUGIN(nifty) {
    py::module niftyModule("nifty", "nifty python bindings");


    py::class_<nifty::NumpyArray<float>>(niftyModule,"NumpyArrayFloat")
    ;

    py::class_<nifty::NumpyArray<uint64_t>>(niftyModule,"NumpyArrayUInt64")
    ;

    //y::implicitly_convertible<py::array_t<float>, nifty::NumpyArray<float> >();
    //py::implicitly_convertible<py::array_t<uint64_t>, nifty::NumpyArray<uint64_t> >();
    using namespace nifty;
    graph::initSubmoduleGraph(niftyModule);
}
